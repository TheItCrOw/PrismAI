import json
import os
from argparse import ArgumentParser, Namespace
from itertools import batched
from pathlib import Path

import pymongo
from bson import DBRef
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm, trange

if Path(".env").exists():
    load_dotenv()
elif Path("../.env").exists():
    load_dotenv("../.env")


def parse_pre_processors(args: Namespace):
    from transition_scores.pre_processor import (
        PreProcessor,
        RollingWindowChunkPreProcessor,
        SlidingWindowTextPreProcessor,
        TextPreProcessor,
    )

    pre_processors: list[PreProcessor] = []
    for pre_processor_str in set(args.pre_processors):
        match pre_processor_str:
            case "TextPreProcessor":
                pre_processors.append(
                    TextPreProcessor.from_pretrained(
                        args.model, max_length=args.max_length
                    )
                )
            case "RollingWindowChunkPreProcessor":
                pre_processors.append(
                    RollingWindowChunkPreProcessor.from_pretrained(
                        args.model, max_length=args.max_length
                    )
                )
            case "SlidingWindowTextPreProcessor":
                pre_processors.append(
                    SlidingWindowTextPreProcessor.from_pretrained(
                        args.model, max_length=args.max_length, stride=args.stride
                    )
                )
            case _:
                raise RuntimeError
    return pre_processors


def parse_scorer_provider(args: Namespace):
    match args.provider:
        case "hf":
            from transition_scores.scorer import TransformersTransitionScorer

            return TransformersTransitionScorer(
                args.model,
                batch_size=args.model_batch_size,
                device=args.device,
                load_in_8bit=args.load_in_8bit,
            )
        case "onnx":
            from transition_scores.scorer import OnnxTransitionScorer

            return OnnxTransitionScorer(
                args.model,
                batch_size=args.model_batch_size,
                device=args.device,
            )
        case _:
            raise RuntimeError


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
        "pre_processors",
        nargs="+",
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

    args = parser.parse_args()

    mongodb_batch_size = args.mongodb_batch_size
    mongodb_filter_query = args.mongodb_filter or {}
    dataset_batch_size = args.dataset_batch_size or mongodb_batch_size

    mongodb_client = MongoClient(args.mongodb_uri)
    mongodb_database = mongodb_client.get_database(args.mongodb_database)
    source_collection = mongodb_database.get_collection(args.source_collection)
    target_collection = mongodb_database.get_collection(args.target_collection)

    scorer = parse_scorer_provider(args)

    pre_processors = parse_pre_processors(args)

    num_documents = args.mongodb_limit or source_collection.count_documents(
        mongodb_filter_query
    )
    tq_fetch = trange(
        args.mongodb_skip,
        args.mongodb_skip + num_documents,
        dataset_batch_size,
        desc=f"Processing Document Batches of {dataset_batch_size} from {args.source_collection}",
    )

    fields_req_by_pre_processors = {
        field
        for pre_processor in pre_processors
        for field in pre_processor.required_fields
    }
    fields_projection = list({"id"} | fields_req_by_pre_processors)
    for offset in tq_fetch:
        dataset = []
        for row in source_collection.find(
            mongodb_filter_query,
            projection=fields_projection,
            batch_size=mongodb_batch_size,
            limit=min(dataset_batch_size, num_documents),
            skip=offset,
        ):
            # Skip rows that do not have all required fields
            if not all(row.get(field, False) for field in fields_req_by_pre_processors):
                continue

            refs = {
                "_ref_id": DBRef(
                    args.source_collection,
                    row.pop("_id"),
                )
            }
            if "id" in row:
                refs["ref_id"] = DBRef(
                    args.source_collection,
                    row.pop("id"),
                )
            else:
                refs["ref_id"] = None

            if "_ref_id" in row:
                refs["_orig_ref_id"] = row.pop("_ref_id")
            if "ref_id" in row:
                refs["orig_ref_id"] = row.pop("ref_id")

            dataset.append(refs | row)

        for pre_processor in pre_processors:
            processed_dataset = scorer.process(dataset, pre_processor)
            for batch in batched(
                tqdm(
                    processed_dataset,
                    desc="Inserting Batch Results",
                    position=1,
                    leave=False,
                ),
                mongodb_batch_size,
            ):
                target_collection.insert_many(batch, ordered=False)
