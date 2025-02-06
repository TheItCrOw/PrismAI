import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from luminar.features import (
    AnyDimFeatures,
    FeatureAlgorithm,
    FeatureSelection,
    OneDimFeatures,
)
from luminar.mongo import MongoDBAdapter
from simple_dataset.dataset import Dataset
from transition_scores.data import TransitionScores


class DocumentClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mongodb_adapter: MongoDBAdapter,
        feature_dim: AnyDimFeatures | tuple[int, ...] = OneDimFeatures(256),
        feature_selection: FeatureSelection.Type | str = FeatureSelection.Type.First,
        feature_type: FeatureAlgorithm.Type | str = FeatureAlgorithm.Type.Likelihood,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.mongodb_adapter = mongodb_adapter
        match feature_selection:
            case string if isinstance(string, str):
                self.feature_selection = FeatureSelection.Type[string].into(feature_dim)
            case value if isinstance(value, FeatureSelection.Type):
                self.feature_selection = value.into(feature_dim)
            case _:
                raise ValueError(
                    f"Unsupported feature selection: {feature_selection}. Must be one of {FeatureSelection.Type.__members__.keys()}"
                )

        match feature_type:
            case string if isinstance(string, str):
                self.feature_algo = FeatureAlgorithm.Type[string].into(
                    self.feature_selection
                )
            case value if isinstance(value, FeatureAlgorithm.Type):
                self.feature_algo = value.into(self.feature_selection)
            case _:
                raise ValueError(
                    f"Unsupported feature type: {feature_type}. Must be one of {FeatureAlgorithm.Type.__members__.keys()}"
                )

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_name_or_path, use_fast=True
        # )

    # def prepare_data(self):
    #     datasets.load_dataset("glue", self.task_name)
    #     AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage=None):
        dataset = Dataset(self.mongodb_adapter.iter_documents())
        # drop documents where we don't have one human and one AI generated variant
        dataset = dataset.filter(
            lambda doc: len(doc["features"]) == 2
            and len({ts["document"]["type"] for ts in doc["features"]}) == 2
        )
        # flatten the nested features list
        dataset = dataset.flat_map(lambda doc: doc["features"])
        # merge strided transition scores
        dataset = dataset.apply(TransitionScores.merge, "transition_scores")

        human_documents: list[TransitionScores] = list(
            # filter for human written documents
            dataset.filter(
                lambda doc: doc["document"]["type"] == "source", in_place=False
            )
            # filter out documents that are too short
            .filter(
                lambda doc: len(doc["transition_scores"])
                >= self.feature_selection.required_size()
            )
            # .map(lambda doc: doc["transition_scores"])
        )
        synth_documents: list[TransitionScores] = list(
            # filter for AI generated documents
            dataset.filter(lambda doc: doc["document"]["type"] != "source")
            # filter out documents that are too short
            .filter(
                lambda doc: len(doc["transition_scores"])
                >= self.feature_selection.required_size()
            )
            # .map(lambda doc: doc["transition_scores"])
        )

        self.eval_data = []
        self.train_data = []

        # featurize transition scores
        with tqdm(total=len(human_documents) + len(synth_documents)) as tq:
            # 80% train, 20% eval split
            eval_indices = np.random.choice(
                len(human_documents), len(human_documents) // 5, replace=False
            )
            eval_indices = sorted(eval_indices, reverse=True)

            for idx in eval_indices:
                document = human_documents.pop(idx)
                self.eval_data.append(
                    {
                        "features": self.convert_to_features(
                            document["transition_scores"]
                        ),
                        "labels": 0,
                    }
                )
                tq.update(1)

            for document in human_documents:
                self.train_data.append(
                    {
                        "features": self.convert_to_features(
                            document["transition_scores"]
                        ),
                        "labels": 0,
                    }
                )
                tq.update(1)

            # 80% train, 20% eval split
            eval_indices = np.random.choice(
                len(synth_documents), len(synth_documents) // 5, replace=False
            )
            eval_indices = sorted(eval_indices, reverse=True)

            for idx in eval_indices:
                document = synth_documents.pop(idx)
                self.eval_data.append(
                    {
                        "features": self.convert_to_features(
                            document["transition_scores"]
                        ),
                        "labels": 1,
                    }
                )
                tq.update(1)

            for document in synth_documents:
                self.train_data.append(
                    {
                        "features": self.convert_to_features(
                            document["transition_scores"]
                        ),
                        "labels": 1,
                    }
                )
                tq.update(1)

    def convert_to_features(self, transition_scores: TransitionScores):
        return self.feature_algo.featurize(transition_scores)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.train_batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.eval_data, batch_size=self.eval_batch_size)

    # def test_dataloader(self):
    #     if len(self.eval_splits) == 1:
    #         return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
    #     elif len(self.eval_splits) > 1:
    #         return [
    #             DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
    #             for x in self.eval_splits
    #         ]
