import json
import os
import sys
import traceback
from argparse import ArgumentParser, Namespace
from itertools import batched
from pathlib import Path

import pymongo
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from tqdm import tqdm

from transition_scores.data import FeaturesDict
from transition_scores.scorer.abc import TransitionScorer, convert_to_mongo

if Path(".env").exists():
    load_dotenv()
elif Path("../.env").exists():
    load_dotenv("../.env")


MAX_TRY_INSERT_RECURSION_DEPTH = os.environ.get("MAX_TRY_INSERT_RECURSION_DEPTH", 16)


def parse_pre_processors(args: Namespace):
    from transition_scores.pre_processor import (
        RollingWindowChunkPreProcessor,
        SlidingWindowTextPreProcessor,
        TruncationTextPreProcessor,
    )

    match args.pre_processor:
        case "TextPreProcessor":
            return TruncationTextPreProcessor.from_pretrained(
                args.model, max_length=args.max_length
            )
        case "RollingWindowChunkPreProcessor":
            return RollingWindowChunkPreProcessor.from_pretrained(
                args.model, max_length=args.max_length
            )
        case "SlidingWindowTextPreProcessor":
            return SlidingWindowTextPreProcessor.from_pretrained(
                args.model, max_length=args.max_length, stride=args.stride
            )
        case _:
            raise RuntimeError


def parse_scorer_model(args: Namespace) -> TransitionScorer:
    match args.provider:
        case "hf":
            from transition_scores.scorer import TransformersTransitionScorer

            return TransformersTransitionScorer(
                args.model,
                batch_size=args.batch_size or args.model_batch_size,
                device=args.device,
                load_in_8bit=args.load_in_8bit,
            )
        case "onnx":
            from transition_scores.scorer import OnnxTransitionScorer

            return OnnxTransitionScorer(
                args.model,
                batch_size=args.batch_size or args.model_batch_size,
                device=args.device,
            )
        case _:
            raise RuntimeError


class TryInsertError(Exception):
    @classmethod
    def from_doc(cls, document: FeaturesDict):
        _ref_id = document.get("refs", {}).get("_ref_id")
        if _ref_id:
            return cls(
                f"Encountered an error while trying to insert document with _ref_id={str(_ref_id)}."
            )
        else:
            return cls()


def try_insert_one(document: FeaturesDict, collection: Collection, recursion_depth=0):
    try:
        mongodb_target_collection.insert_one(document)
    except pymongo.errors.DuplicateKeyError:
        return  # ignore duplicates
    except pymongo.errors.DocumentTooLarge:
        pass  # document still too large, variant 1
    except pymongo.errors.WriteError:
        pass  # document still too large, variant 2
    except Exception as e:
        if str(e).strip == "ValueError: Document would overflow BSON size limit":
            pass  # document still too large, variant 3
        else:
            # otherwise, raise the error
            raise TryInsertError.from_doc(document) from e
    else:
        return

    if recursion_depth < MAX_TRY_INSERT_RECURSION_DEPTH:
        for document_split in document.split():
            return try_insert_one(document_split, collection, recursion_depth + 1)
    else:
        raise RecursionError(
            f"Maximum recursion depth ({MAX_TRY_INSERT_RECURSION_DEPTH}) exceeded; aborting insert."
            "Consider increasing MAX_TRY_INSERT_RECURSION_DEPTH."
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("model", type=str, help="Model name or path")
    provider_group_me = model_group.add_mutually_exclusive_group()
    provider_group_me.add_argument(
        "--provider",
        choices=["hf", "onnx"],
        help="Model execution provider. Either `hf` for huggingface transformers or `onnx` for the ONNX Runtime via optimum.",
        default="hf",
    )
    provider_group_me.add_argument(
        "--hf",
        action="store_const",
        const="hf",
        dest="provider",
        help="Use huggingface transformers as the model provider.",
    )
    provider_group_me.add_argument(
        "--onnx",
        action="store_const",
        const="onnx",
        dest="provider",
        help="Use ONNX Runtime via optimum as the model provider.",
    )

    model_group.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load huggingface model in 8-bit precision using bitsandbytes.",
    )
    model_group.add_argument(
        "-bsm",
        "--model_batch_size",
        type=int,
        metavar="N",
        default=32,
    )
    model_group.add_argument(
        "-bsd",
        "--dataset_batch_size",
        type=int,
        default=128,
        metavar="N",
        help="Batch size for dataset processing. Defaults to the MongoDB batch size.",
    )

    device_group = model_group.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        type=str,
        help="Device to use",
        default=None,
    )
    device_group.add_argument(
        "--cpu",
        action="store_const",
        const="cpu",
        dest="device",
        help="Use the CPU.",
    )
    device_group.add_argument(
        "--cuda",
        "--gpu",
        action="store_const",
        const="cuda",
        dest="device",
        help="Use CUDA GPUs.",
    )

    group_pre_processor = parser.add_argument_group("Pre-Processors")
    group_pre_processor.add_argument(
        "pre_processor",
        choices=[
            "RollingWindowChunkPreProcessor",
            "SlidingWindowTextPreProcessor",
            "TextPreProcessor",
        ],
        help="Pre-processor to use.",
    )
    group_pre_processor.add_argument(
        "-tp",
        "--text_pre_processor",
        action="append_const",
        const="TextPreProcessor",
        help="Use the TextPreProcessor.",
    )
    group_pre_processor.add_argument(
        "-rwc",
        "--rolling_window_chunks",
        action="append_const",
        const="RollingWindowChunkPreProcessor",
        help="Use the RollingWindowChunkPreProcessor.",
    )
    group_pre_processor.add_argument(
        "-slt",
        "--sliding_window_text",
        action="append_const",
        const="SlidingWindowTextPreProcessor",
        help="Use the SlidingWindowTextPreProcessor.",
    )
    group_pre_processor.add_argument(
        "--max_length",
        type=int,
        metavar="N",
        help="Maximum length of the tokenized sequences. Unless specified, will use the model's maximum input size.",
        default=None,
    )
    group_pre_processor.add_argument(
        "--stride",
        type=int,
        metavar="N",
        help="SlidingWindowTextPreProcessor: set the stride for the sliding window. Defaults to 1/4 of the max_length.",
        default=None,
    )

    mongodb_group = parser.add_argument_group("MongoDB")
    mongodb_group.add_argument(
        "--filter",
        type=json.loads,
        metavar="{...}",
        dest="mongodb_filter",
        help="Filter for MongoDB query as JSON string.",
        default=None,
    )
    mongodb_group.add_argument(
        "--domains",
        type=str,
        nargs="+",
        dest="mongodb_filter_domains",
        help="Filter by domain. Overwrites the `filter` argument.",
        default=None,
    )
    mongodb_group.add_argument(
        "--uri",
        "--mongodb_uri",
        type=str,
        metavar="mongodb://...",
        dest="mongodb_uri",
        default=os.environ.get("MONGO_DB_CONNECTION", None),
        help="MongoDB connection URI. Defaults to the `MONGO_DB_CONNECTION` environment variable if set.",
    )
    mongodb_group.add_argument(
        "-bsdb",
        "--mongodb_batch_size",
        type=int,
        metavar="N",
        default=512,
        help="Batch size for MongoDB query.",
    )
    mongodb_group.add_argument(
        "-db",
        "--database",
        type=str,
        metavar="NAME",
        dest="mongodb_database",
        default="prismai",
        help="MongoDB database name.",
    )
    mongodb_group.add_argument(
        "-sc",
        type=str,
        metavar="NAME",
        dest="source_collection",
        default="collected_items",
        help="Source collection name.",
    )
    mongodb_group.add_argument(
        "-tc",
        type=str,
        metavar="NAME",
        dest="target_collection",
        default="transition_scores",
        help="Target collection name.",
    )
    mongodb_group.add_argument(
        "--limit",
        type=int,
        metavar="N",
        dest="mongodb_limit",
        default=None,
        help="Limit query to N documents.",
    )
    mongodb_group.add_argument(
        "--skip",
        type=int,
        metavar="N",
        dest="mongodb_skip",
        default=0,
        help="Skip the first N documents.",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=None,
        help="Global batch size override.",
    )

    args = parser.parse_args()

    mongodb_batch_size = args.batch_size or args.mongodb_batch_size
    mongodb_filter_query = args.mongodb_filter or {}
    dataset_batch_size = (
        args.batch_size or args.dataset_batch_size or mongodb_batch_size
    )
    dataset_batch_size = min(dataset_batch_size, mongodb_batch_size)

    mongodb_client = MongoClient(args.mongodb_uri)
    mongodb_database = mongodb_client.get_database(args.mongodb_database)
    mongodb_source_collection = mongodb_database.get_collection(args.source_collection)
    mongodb_target_collection = mongodb_database.get_collection(args.target_collection)

    model = parse_scorer_model(args)
    model_metadata = model.get_metadata()

    pre_processor = parse_pre_processors(args)
    pre_processor_metadata = pre_processor.get_metadata()

    fields_projection = list(
        {"id", "_ref_id", "ref_id"} | set(pre_processor.required_fields.keys())
    )

    domains = args.mongodb_filter_domains or mongodb_filter_query.pop("domain", (None,))

    for domain in tqdm(domains, position=0, desc="Processing Domains"):
        if domain:
            mongodb_filter_query["domain"] = domain

        mongodb_limit = args.mongodb_limit or mongodb_source_collection.count_documents(
            mongodb_filter_query
        )

        for dataset in batched(
            tqdm(
                mongodb_source_collection.find(
                    mongodb_filter_query,
                    projection=fields_projection,
                    batch_size=mongodb_batch_size,
                    limit=mongodb_limit,
                    skip=args.mongodb_skip,
                ),
                total=mongodb_limit,
                desc=f"Processing Documents from {domain or 'All Domains'}",
                position=1,
                leave=False,
            ),
            dataset_batch_size,
        ):
            dataset = pre_processor.pre_process(dataset)
            scores = model.process(dataset, pre_processor.pad_token_id)
            dataset = pre_processor.post_process(dataset, scores)

            for batch in batched(
                tqdm(
                    dataset,
                    desc="Inserting Document Batches",
                    position=2,
                    leave=False,
                ),
                mongodb_batch_size,
            ):
                batch = [
                    convert_to_mongo(
                        document,
                        args.source_collection,
                        model_metadata,
                        pre_processor_metadata,
                    )
                    for document in batch
                ]
                try:
                    mongodb_target_collection.insert_many(batch)
                except Exception as e:
                    err_str = str(e)
                    err_str = err_str[:44] + "..." if len(err_str) > 48 else err_str
                    # print(
                    #     f"Caught error during insert_many: {err_str}"
                    #     f" - attempting insert_one for {len(batch)} documents.",
                    #     file=sys.stderr,
                    # )
                    for document in tqdm(
                        batch,
                        position=2,
                        leave=False,
                        desc="Inserting Documents Individually",
                    ):
                        try:
                            try_insert_one(document, mongodb_target_collection)
                        except TryInsertError as e:
                            traceback.print_exc()
                            continue
                        except RecursionError as e:
                            print(str(e), file=sys.stderr)
                            continue
