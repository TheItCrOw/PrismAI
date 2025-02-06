from hashlib import sha256
from pathlib import Path
from typing import Generator, Literal

import bson.json_util
from pymongo import MongoClient
from pymongo.database import Database
from tqdm import tqdm


class MongoDBAdapter:
    __client: MongoClient
    __db: Database

    def __init__(
        self,
        mongo_db_connection: str,
        database: str = "prismai",
        source_collection: str = "collected_items",
        synth_collection: str = "synthesized_texts",
        score_collection: str = "transition_scores",
        source_collection_limit: int = 1500,
        domain: str = "bundestag",
        lang: str | None = None,
        # pre_processor_type: Literal[""] = "sliding-window",
        synth_type: Literal["fulltext", "chunk"] = "fulltext",
        synth_agent: Literal["gemma2:9b", "nemotron", "gpt-4o-mini"] = "gpt-4o-mini",
        feature_model: Literal["gpt2", "meta-llama/Llama-3.2-1B"] = "gpt2",
        additional_source_match: dict | None = None,
        additional_synth_match: dict | None = None,
        additional_score_match: dict | None = None,
        additional_pipeline_stages: list[dict] | None = None,
    ):
        self.__client = MongoClient(mongo_db_connection)
        self.__db = self.__client.get_database(database)

        self.source_collection = source_collection
        self.source_collection_limit = source_collection_limit
        self.synth_collection = synth_collection
        self.score_collection = score_collection
        self.domain = domain
        self.lang = lang or ("en-EN" if self.domain != "bundestag" else "de-DE")
        self.synth_type = synth_type
        self.synth_agent = synth_agent
        self.feature_model = feature_model
        self.additional_source_match = additional_source_match or {}
        self.additional_synth_match = additional_synth_match or {}
        self.additional_score_match = additional_score_match or {}
        self.additional_pipeline_stages = additional_pipeline_stages or []

    def _get_aggregation_pipeline(self):
        return [
            {
                "$match": {"domain": self.domain, "lang": self.lang}
                | self.additional_source_match
            },
            {"$limit": self.source_collection_limit},
            {
                "$lookup": {
                    "from": self.synth_collection,
                    "as": "synthesized_texts",
                    "localField": "_id",
                    "foreignField": "_ref_id.$id",
                    "pipeline": [
                        {
                            "$match": {
                                "type": self.synth_type,
                                "agent": self.synth_agent,
                            }
                            | self.additional_synth_match
                        },
                    ],
                }
            },
            {
                "$lookup": {
                    "from": self.score_collection,
                    "as": "features",
                    "localField": "_id",
                    "foreignField": "document._id.$id",
                    "pipeline": [
                        {
                            "$match": {
                                "model.name": self.feature_model,
                                # "pre_processor.type": pre_processor_type,
                                "document.type": {"$in": ["source", self.synth_type]},
                                "document.agent": {"$in": [None, self.synth_agent]},
                            }
                            | self.additional_score_match,
                        },
                    ],
                },
            },
        ] + self.additional_pipeline_stages

    def iter_documents(self) -> Generator[dict, None, None]:
        # if synth_type == "chunk":
        #     raise NotImplementedError("Chunk")

        yield from tqdm(
            self.__db.get_collection(self.source_collection).aggregate(
                self._get_aggregation_pipeline(),
                allowDiskUse=True,
            ),
            desc=f"{type(self).__name__}: Loading Documents from MongoDB",
            total=self.source_collection_limit,
        )

    def get_documents(self) -> list[dict]:
        return list(self.iter_documents())

    def _get_pipeline_hash(self) -> str:
        return sha256(
            bson.json_util.dumps(self._get_aggregation_pipeline()).encode()
        ).hexdigest()

    def get_cache_file(self, ext="json") -> Path:
        return Path("/tmp/luminar") / f"{self._get_pipeline_hash()}.{ext}"
