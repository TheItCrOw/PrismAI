import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from luminar.features import (
    FeatureSelection,
    FeatureType,
    LogLikelihoodLogRankRatio,
    OneDimFeatures,
    ThreeDimFeatures,
    TwoDimFeatures,
)
from luminar.mongo import MongoDBAdapter
from simple_dataset.dataset import Dataset
from transition_scores.data import TransitionScores


class DocumentClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        mongodb_adapter: MongoDBAdapter,
        slice: OneDimFeatures | TwoDimFeatures | ThreeDimFeatures = OneDimFeatures(256),
        feature_selection: type[FeatureSelection] = None,
        feature_type: str | FeatureType = FeatureType.LLR,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.mongodb_adapter = mongodb_adapter
        feature_selection = (
            feature_selection(slice)
            if feature_selection is not None
            else FeatureSelection.first(slice)
        )
        match feature_type:
            case FeatureType.LLR | "LLR" | "llr":
                self.feature_algo = LogLikelihoodLogRankRatio(feature_selection)
            case FeatureType.LTR | "LTR" | "ltr":
                raise NotImplementedError
            case FeatureType.LTS | "LTS" | "lts":
                raise NotImplementedError
            case _:
                raise RuntimeError(
                    f"Unsupported feature type: {feature_type}. Must be one of {FeatureType.__members__.keys()}"
                )

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_name_or_path, use_fast=True
        # )

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
            dataset.filter(
                lambda doc: doc["document"]["type"] == "source", in_place=False
            ).map(lambda doc: doc["transition_scores"], in_place=True)
        )
        synth_documents: list[TransitionScores] = list(
            dataset.filter(
                lambda doc: doc["document"]["type"] != "source", in_place=True
            ).map(lambda doc: doc["transition_scores"], in_place=True)
        )

        eval_indices = np.random.choice(
            len(human_documents), len(human_documents) // 5, replace=False
        )
        eval_indices = sorted(eval_indices, reverse=True)

        self.eval_data = []
        for idx in eval_indices:
            document = human_documents.pop(idx)
            self.eval_data.append((self.convert_to_features(document), 0))

        for idx in eval_indices:
            document = synth_documents.pop(idx)
            self.eval_data.append((self.convert_to_features(document), 1))

        self.train_data = []
        for document in human_documents:
            self.train_data.append((self.convert_to_features(document), 0))

        for document in synth_documents:
            self.train_data.append((self.convert_to_features(document), 1))

    def convert_to_features(self, transition_scores: TransitionScores):
        return self.feature_algo.featurize(transition_scores)

    # def prepare_data(self):
    #     datasets.load_dataset("glue", self.task_name)
    #     AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

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
