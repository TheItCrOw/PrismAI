import json
import os
from argparse import ArgumentParser
from itertools import batched
from pathlib import Path

import datasets
from datasets import Dataset
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

from transition_scores.pre_processor.abc import PreProcessor
from transition_scores.pre_processor.chunks import (
    RollingWindowChunkPreProcessor,
)
from transition_scores.pre_processor.text import TextPreProcessor
from transition_scores.scorer import OnnxTransitionScorer, TransformersTransitionScorer

if Path(".env").exists():
    load_dotenv()
elif Path("../.env").exists():
    load_dotenv("../.env")

datasets.disable_progress_bars()

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

    args = parser.parse_args()

    mongodb_client = MongoClient(args.mongodb_uri)
    mongodb_database = mongodb_client.get_database(args.mongodb_database)
    source_collection = mongodb_database.get_collection(args.source_collection)
    num_documents = source_collection.count_documents(args.mongodb_filter)

    target_collection = mongodb_database.get_collection(args.target_collection)

    match args.provider:
        case "hf":
            scorer = TransformersTransitionScorer(
                args.model,
                batch_size=args.model_batch_size,
                device=args.device,
            )
        case "onnx":
            scorer = OnnxTransitionScorer(
                args.model,
                batch_size=args.model_batch_size,
                device=args.device,
            )
        case _:
            raise RuntimeError

    pre_processors: list[PreProcessor] = []
    for pre_processor_str in args.pre_processors:
        match pre_processor_str:
            case "TextPreProcessor":
                pre_processors.append(TextPreProcessor.from_pretrained(args.model))
            case "RollingWindowChunkPreProcessor":
                pre_processors.append(
                    RollingWindowChunkPreProcessor.from_pretrained(args.model)
                )
            case _:
                raise RuntimeError

    mongodb_batch_size = args.mongodb_batch_size
    dataset_batch_size = args.dataset_batch_size or mongodb_batch_size

    tq_fetch = tqdm(
        source_collection.find(
            projection=[
                "text",
                "chunks",
            ],
            batch_size=mongodb_batch_size,
        ),
        total=num_documents,
        desc=f"Processing Documents from {args.source_collection}",
    )
    for batch in batched(tq_fetch, dataset_batch_size):
        batch = [
            {
                "source": {
                    "$ref": args.source_collection,
                    "$id": str(row.pop("_id")),
                }
            }
            | row
            for row in batch
        ]
        dataset = Dataset.from_list(batch)
        dataset = dataset.filter(lambda x: x["text"] and x["chunks"])

        for pre_processor in pre_processors:
            scorer.set_pre_processor(pre_processor)
            processed_dataset = scorer.process(dataset)
            target_collection.insert_many(
                tqdm(
                    processed_dataset.to_list(),
                    desc="Inserting Batch Results",
                    position=1,
                    leave=False,
                ),
                ordered=False,
            )
