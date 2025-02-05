from typing import Generator, Literal

from pymongo import MongoClient
from pymongo.database import Database


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
        self.lang = lang
        self.synth_type = synth_type
        self.synth_agent = synth_agent
        self.feature_model = feature_model
        self.additional_source_match = additional_source_match or {}
        self.additional_synth_match = additional_synth_match or {}
        self.additional_score_match = additional_score_match or {}
        self.additional_pipeline_stages = additional_pipeline_stages or []

    def iter_documents(self) -> Generator[dict, None, None]:
        # if synth_type == "chunk":
        #     raise NotImplementedError("Chunk")

        lang = self.lang or ("en-EN" if self.domain != "bundestag" else "de-DE")

        yield from self.__db.get_collection(self.source_collection).aggregate(
            [
                {
                    "$match": {"domain": self.domain, "lang": lang}
                    | self.additional_source_match
                },
                {"$limit": self.source_collection_limit},
                {"$project": {"_id": 1}},
                {
                    "$lookup": {
                        "from": self.synth_collection,
                        "as": "synthesized_texts",
                        "localField": "_id",
                        "foreignField": "_ref_id.$id",
                        "pipeline": [
                            # {
                            #     "$project": {"_id": 1, "type": 1, "agent": 1}
                            #     | {key: 1 for key in additional_synth_match}
                            # },
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
                            # {
                            #     "$project": {
                            #         "_id": 1,
                            #         "document": 1,
                            #         "model.name": 1,
                            #         # "pre_processor.type": 1,
                            #         "transition_scores": 1,
                            #         "metadata": 1,
                            #     }
                            #     | {key: 1 for key in additional_score_match}
                            # },
                            {
                                "$match": {
                                    "model.name": self.feature_model,
                                    # "pre_processor.type": pre_processor_type,
                                    "document.type": {
                                        "$in": ["source", self.synth_type]
                                    },
                                    "document.agent": {"$in": [None, self.synth_agent]},
                                }
                                | self.additional_score_match,
                            },
                        ],
                    },
                },
            ]
            + self.additional_pipeline_stages,
            allowDiskUse=True,
        )

    def get_documents(self) -> list[dict]:
        return list(self.iter_documents())
