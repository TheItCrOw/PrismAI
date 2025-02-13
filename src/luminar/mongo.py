import pickle
from hashlib import sha256
from pathlib import Path
from typing import Final, Generator, Literal, Self

import bson.json_util
from pymongo import MongoClient
from pymongo.collection import Collection as MongoCollection
from pymongo.database import Database as MongoDatabase
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


DEFAULT_CACHE_DIR: Final[Path] = (
    Path("/nvme/.cache/luminar") if Path("/nvme").exists() else Path("/tmp/luminar")
)


class MongoDataset(TorchDataset):
    def __init__(
        self,
        mongo_db_connection: str,
        collection: str,
        pipeline: list[dict] = [],
        database: str = "prismai",
        use_cache: bool = True,
        cache_dir: Path | None = None,
        update_cache: bool = False,
    ):
        super().__init__()
        self.mongo_db_connection = mongo_db_connection
        self.database = database
        self.collection = collection
        self.pipeline = pipeline
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.update_cache = update_cache

    @property
    def data(self) -> list[list[dict]]:
        self.load()
        return self._data

    def load(self) -> Self:
        if hasattr(self, "_data"):
            return self

        cache_file = self.get_cache_file()

        if self.use_cache and not self.update_cache and cache_file.exists():
            print(f"[{type(self).__name__}] Loading Data from Cache File {cache_file}")
            with cache_file.open("rb") as fp:
                self._data = pickle.load(fp)
        else:
            with MongoClient(self.mongo_db_connection) as client:
                db: MongoDatabase = client.get_database(self.database)
                collection: MongoCollection = db.get_collection(self.collection)

                self._data: list[list[dict]] = list(
                    tqdm(
                        collection.aggregate(
                            self.pipeline,
                            allowDiskUse=True,
                        ),
                        desc=f"[{type(self).__name__}] Loading Documents from MongoDB",
                    )
                )

        if self.use_cache and self.update_cache or not cache_file.exists():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{type(self).__name__}] Writing Cache File {cache_file}")
            with cache_file.open("wb") as fp:
                pickle.dump(self._data, fp)

        return self

    def get_cache_file(self) -> Path:
        pipeline_hash = sha256(self.collection.encode())
        pipeline_hash.update(bson.json_util.dumps(self.pipeline).encode())
        cache_file = self.cache_dir / f"{pipeline_hash.hexdigest()}.pkl"
        return cache_file

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> list[dict]:
        return self.data[idx]

    def __iter__(self) -> Generator[dict, None, None]:
        return iter(self.data)


class PrismaiDataset(MongoDataset):
    def __init__(
        self,
        mongo_db_connection: str,
        database: str = "prismai",
        collection: str = "features_prismai",
        domain: str = "bundestag",
        split: str = "train",
        lang: str | None = None,
        # pre_processor_type: Literal[""] = "sliding-window",
        document_type: Literal["fulltext", "chunk"] = "fulltext",
        synth_agent: Literal["gemma2:9b", "nemotron", "gpt-4o-mini"] = "gpt-4o-mini",
        feature_model: Literal["gpt2", "meta-llama/Llama-3.2-1B"] = "gpt2",
        additional_match_conditions: dict | None = None,
        additional_pipeline_stages: list[dict] | None = None,
        use_cache: bool = True,
        cache_dir: Path | None = None,
        update_cache: bool = False,
    ):
        additional_match_conditions = additional_match_conditions or {}
        additional_pipeline_stages = additional_pipeline_stages or []

        match_filter = {}
        if synth_agent:
            if isinstance(synth_agent, str):
                match_filter["document.agent"] = {"$in": [None, synth_agent]}
            else:
                match_filter["document.agent"] = synth_agent
        if domain:
            match_filter["document.domain"] = domain
            match_filter["document.lang"] = lang or (
                "en-EN" if domain not in {"bundestag", "spiegel_articles"} else "de-DE"
            )
        elif lang:
            match_filter["document.lang"] = lang
        if document_type:
            if isinstance(synth_agent, str):
                match_filter["document.type"] = {"$in": ["source", document_type]}
        if feature_model:
            match_filter["model.name"] = feature_model
        if split:
            match_filter["split"] = split
        # match_filter |= {
        #     # "pre_processor.type": pre_processor_type,
        # }
        match_filter |= additional_match_conditions

        pipeline = [
            {"$match": match_filter},
            {
                "$group": {
                    "_id": "$document._id.$id",
                    "features": {
                        "$push": {
                            "label": "$document.label",
                            "type": "$document.type",
                            "split": "$split",
                            "transition_scores": "$transition_scores",
                        }
                    },
                }
            },
        ] + additional_pipeline_stages

        super().__init__(
            mongo_db_connection=mongo_db_connection,
            collection=collection,
            pipeline=pipeline,
            database=database,
            use_cache=use_cache,
            cache_dir=cache_dir,
            update_cache=update_cache,
        )
