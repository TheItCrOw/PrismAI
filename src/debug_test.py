import os
from itertools import batched

import datasets
from datasets import Dataset
from pymongo import MongoClient
from tqdm import tqdm, trange

from transition_scores.pre_processor.chunks import RollingWindowChunkPreProcessor
from transition_scores.scorer import OnnxTransitionScorer

mongodb_batch_size = 16
mongodb_filter_query = {}
dataset_batch_size = 32

mongodb_client = MongoClient(os.environ.get("MONGO_DB_CONNECTION"))
mongodb_database = mongodb_client.get_database("prismai")
source_collection = mongodb_database.get_collection("collected_items")
target_collection = mongodb_database.get_collection("test")


scorer = OnnxTransitionScorer(
    "/hot_storage/models/onnx/gpt2_onnx_o4/",
    batch_size=4,
    device="cuda",
)

pre_processors = RollingWindowChunkPreProcessor.from_pretrained("gpt2")
num_documents = source_collection.count_documents(mongodb_filter_query)
tq_fetch = trange(
    0,
    128,
    32,
    desc=f"Processing Document Batches of {dataset_batch_size} from {source_collection}",
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
                "$ref": "collected_items",
                "$id": str(row.pop("_id")),
            }
        }
        if "id" in row:
            refs["ref_id"] = {
                "$ref": "collected_items",
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

    for pre_processor in [pre_processors]:
        processed_dataset = scorer.process(dataset, pre_processor)
        for r_batch in batched(
            tqdm(
                processed_dataset,
                desc="Inserting Batch Results",
                position=1,
                leave=False,
            ),
            mongodb_batch_size,
        ):
            target_collection.insert_many(r_batch, ordered=False)
