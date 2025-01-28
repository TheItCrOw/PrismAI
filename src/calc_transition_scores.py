import json
import os
from argparse import ArgumentParser, Namespace
from itertools import batched
from pathlib import Path

import datasets
from datasets import Dataset
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
        TextPreProcessor,
    )

    pre_processors: list[PreProcessor] = []
    for pre_processor_str in set(args.pre_processors):
        match pre_processor_str:
            case "TextPreProcessor":
                pre_processors.append(TextPreProcessor.from_pretrained(args.model))
            case "RollingWindowChunkPreProcessor":
                pre_processors.append(
                    RollingWindowChunkPreProcessor.from_pretrained(args.model)
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
    )
    provider_group_me.add_argument(
        "--onnx",
        action="store_const",
        const="onnx",
        dest="provider",
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
        default=32,
    )
    model_group.add_argument(
        "-bsd",
        "--dataset_batch_size",
        default=None,
        help="Batch size for dataset processing. Defaults to the MongoDB batch size.",
    )

    device_group = model_group.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        type=str,
        help="Device to use",
        default=None,
    )
    device_group.add_argument("--cpu", action="store_const", const="cpu", dest="device")
    device_group.add_argument(
        "--cuda", "--gpu", action="store_const", const="cuda", dest="device"
    )

    group_pre_processor = parser.add_argument_group("Pre-Processors")
    group_pre_processor.add_argument(
        "pre_processors",
        nargs="+",
        choices=[
            "TextPreProcessor",
            "RollingWindowChunkPreProcessor",
        ],
    )
    group_pre_processor.add_argument(
        "-tp",
        "--text_pre_processor",
        action="append_const",
        const="TextPreProcessor",
    )
    group_pre_processor.add_argument(
        "-rwc",
        "--rolling_window_chunks",
        action="append_const",
        const="RollingWindowChunkPreProcessor",
    )
    group_pre_processor.add_argument(
        "--max_length",
        type=int,
        help="Maximum length of the tokenized sequences. Unless specified, will use the model's maximum input size.",
        default=None,
    )
    # group_pre_processor.add_argument(
    #     "--skip_prefix_tokens",
    #     type=int,
    #     help="Skip the first n tokens of the input sequence.",
    # )

    mongodb_group = parser.add_argument_group("MongoDB")
    mongodb_group.add_argument(
        "--filter",
        dest="mongodb_filter",
        type=json.loads,
        help="Filter for MongoDB query as JSON string.",
        default=None,
    )
    mongodb_group.add_argument(
        "--uri",
        "--mongodb_uri",
        dest="mongodb_uri",
        type=str,
        default=os.environ.get("MONGO_DB_CONNECTION", None),
        help="MongoDB connection URI. Defaults to the `MONGO_DB_CONNECTION` environment variable if set.",
    )
    mongodb_group.add_argument(
        "-bsdb",
        "--mongodb_batch_size",
        type=int,
        default=128,
        help="Batch size for MongoDB query.",
    )
    mongodb_group.add_argument(
        "-db",
        "--database",
        dest="mongodb_database",
        type=str,
        default="prismai",
    )
    mongodb_group.add_argument(
        "-sc",
        dest="source_collection",
        type=str,
        default="collected_items",
    )
    mongodb_group.add_argument(
        "-tc",
        "--target_collection",
        dest="target_collection",
        type=str,
        default="transition_scores",
    )
    mongodb_group.add_argument(
        "--limit",
        dest="mongodb_limit",
        type=int,
        default=None,
    )
    mongodb_group.add_argument(
        "--skip",
        dest="mongodb_skip",
        type=int,
        default=0,
    )

    datasets_group = parser.add_argument_group("Datasets")
    datasets_group.add_argument(
        "--enable_progress_bars",
        action="store_true",
        help="enable tqdm progress bars for datasets.",
    )
    datasets_group.add_argument(
        "--enable_cache",
        action="store_true",
        help="Enable caching for datasets.",
    )

    args = parser.parse_args()

    # if not args.enable_progress_bars:
    #     datasets.disable_progress_bars()
    # if not args.enable_cache:
    #     datasets.disable_caching()
    #     # datasets.config.IN_MEMORY_MAX_SIZE = 32 * 1024**2

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
        desc=f"Processing Documents from {args.source_collection}",
    )
    for offset in tq_fetch:
        batch = []
        for row in source_collection.find(
            mongodb_filter_query,
            projection=[
                "text",
                "chunks",
                "id",
            ],
            batch_size=mongodb_batch_size,
            limit=min(dataset_batch_size, num_documents),
            skip=offset,
        ):
            refs = {
                "_ref_id": {
                    "$ref": args.source_collection,
                    "$id": str(row.pop("_id")),
                }
            }
            if "id" in row:
                refs["ref_id"] = {
                    "$ref": args.source_collection,
                    "$id": row.pop("id"),
                }
            else:
                refs["ref_id"] = None

            if "_ref_id" in row:
                refs["_orig_ref_id"] = row.pop("_ref_id")
            if "ref_id" in row:
                refs["orig_ref_id"] = row.pop("ref_id")

            batch.append(refs | row)
        dataset = Dataset.from_list(batch).filter(
            lambda x: x["text"] and x["chunks"],
            keep_in_memory=not datasets.is_caching_enabled(),
        )

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
                target_collection.insert_many(
                    batch,
                    ordered=False,
                )
