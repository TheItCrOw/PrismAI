import pickle
from hashlib import sha256
from pathlib import Path
from typing import Generator, Literal, Self

import bson.json_util
from pymongo import MongoClient
from pymongo.collection import Collection as MongoCollection
from pymongo.database import Database as MongoDatabase
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


class MongoDatset(TorchDataset):
    def __init__(
        self,
        mongo_db_connection: str,
        database: str = "prismai",
        collection: str = "features_prismai",
        domain: str = "bundestag",
        split: str = "train",
        lang: str | None = None,
        # pre_processor_type: Literal[""] = "sliding-window",
        synth_type: Literal["fulltext", "chunk"] = "fulltext",
        synth_agent: Literal["gemma2:9b", "nemotron", "gpt-4o-mini"] = "gpt-4o-mini",
        feature_model: Literal["gpt2", "meta-llama/Llama-3.2-1B"] = "gpt2",
        additional_match_conditions: dict | None = None,
        additional_pipeline_stages: list[dict] | None = None,
        use_cache: bool = True,
        cache_dir: Path = Path("/tmp/luminar"),
        update_cache: bool = False,
    ):
        self.mongo_db_connection = mongo_db_connection
        self.database = database
        self.feature_collection = collection

        lang = lang or (
            "en-EN" if domain not in {"bundestag", "spiegel_articles"} else "de-DE"
        )
        additional_match_conditions = additional_match_conditions or {}
        additional_pipeline_stages = additional_pipeline_stages or []

        self.aggregation_pipeline = [
            {
                "$match": {
                    "document.agent": {"$in": [None, synth_agent]},
                    "document.domain": domain,
                    "document.lang": lang,
                    "document.type": {"$in": ["source", synth_type]},
                    "model.name": feature_model,
                    "split": split,
                    # "pre_processor.type": pre_processor_type,
                }
                | additional_match_conditions
            },
            {
                "$group": {
                    "_id": "$document._id.$id",
                    "features": {
                        "$push": {
                            "type": "$document.type",
                            "split": "$split",
                            "transition_scores": "$transition_scores",
                        }
                    },
                }
            },
        ] + additional_pipeline_stages

        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.update_cache = update_cache

    @property
    def data(self) -> list[list[dict]]:
        self.load()
        return self._data

    def load(self) -> Self:
        if hasattr(self, "_data"):
            return self

        pipeline_hash = sha256(self.feature_collection.encode())
        pipeline_hash.update(bson.json_util.dumps(self.aggregation_pipeline).encode())
        cache_file = self.cache_dir / f"{pipeline_hash.hexdigest()}.pkl"

        if self.use_cache and not self.update_cache and cache_file.exists():
            print(f"[{type(self).__name__}] Loading Data from Cache File {cache_file}")
            with cache_file.open("rb") as fp:
                self._data = pickle.load(fp)
        else:
            with MongoClient(self.mongo_db_connection) as client:
                db: MongoDatabase = client.get_database(self.database)
                collection: MongoCollection = db.get_collection(self.feature_collection)

                self._data: list[list[dict]] = list(
                    tqdm(
                        collection.aggregate(
                            self.aggregation_pipeline,
                            allowDiskUse=True,
                        ),
                        desc=f"[{type(self).__name__}] Loading Documents from MongoDB",
                    )
                )

        if self.use_cache and self.update_cache or not cache_file.exists():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as fp:
                pickle.dump(
                    tqdm(
                        self._data,
                        desc=f"[{type(self).__name__}] Writing Cache File {cache_file}",
                    ),
                    fp,
                )

        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> list[dict]:
        return self.data[idx]

    def __iter__(self) -> Generator[dict, None, None]:
        return iter(self.data)
