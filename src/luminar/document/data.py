import pickle

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from luminar.features import (
    AnyDimFeatures,
    FeatureAlgorithm,
    FeatureSelection,
    OneDimFeatures,
)
from luminar.mongo import MongoDBAdapter
from simple_dataset.dataset import Dataset
from transition_scores.data import TransitionScores


class DocumentClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        mongodb_adapter: MongoDBAdapter,
        feature_dim: AnyDimFeatures | tuple[int, ...] = OneDimFeatures(256),
        feature_selection: FeatureSelection.Type | str = FeatureSelection.Type.First,
        feature_type: FeatureAlgorithm.Type | str = FeatureAlgorithm.Type.Likelihood,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_split_size: float = 0.2,
        use_cache: bool = True,
        update_cache: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(
            {
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "eval_split_size": eval_split_size,
                "feature_dim": feature_dim,
                "feature_selection": str(feature_selection),
                "feature_type": str(feature_type),
            }
        )

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
        self.eval_split_size = eval_split_size

        self._use_cache = use_cache
        self._update_cache = update_cache

    def prepare_data(self):
        cache_file = self.mongodb_adapter.get_cache_file("pkl")
        if self._use_cache:
            if self._update_cache or not cache_file.exists():
                print("Caching Enabled - Loading Dataset in prepare_data()")
                dataset = Dataset(self.mongodb_adapter.iter_documents())

                print(f"Writing Dataset to {cache_file}")
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(dataset.data, f)
            else:
                print("Caching Enabled - Dataset Already Cached")
        else:
            self._dataset = Dataset(self.mongodb_adapter.iter_documents())

    def setup(self, stage=None):
        match stage:
            case "fit" if getattr(self, "train_data", None) is not None:
                return
            case "validation" if getattr(self, "eval_data", None) is not None:
                return
            case "test" if getattr(self, "test_data", None) is not None:
                return
            case "predict":
                raise NotImplementedError(stage)
            case None:
                pass

        if (
            getattr(self, "human_features", None) is None
            or getattr(self, "synth_features", None) is None
        ):
            if self._use_cache:
                print(f"Caching Enabled - Loading Dataset in setup({stage})")
                try:
                    with self.mongodb_adapter.get_cache_file("pkl").open("rb") as f:
                        self._dataset = Dataset(pickle.load(f))
                except FileNotFoundError as e:
                    raise RuntimeError(
                        "Cache file not found: did you forget to run prepare_data() with use_cache=True?"
                    ) from e
            else:
                if getattr(self, "_dataset", None) is None:
                    self._dataset = Dataset(self.mongodb_adapter.iter_documents())

            # drop documents where we don't have one human and one AI generated variant
            self._dataset = self._dataset.filter(
                lambda doc: len(doc["features"]) == 2
                and len({ts["document"]["type"] for ts in doc["features"]}) == 2
            )
            # flatten the nested features list
            self._dataset = self._dataset.flat_map(lambda doc: doc["features"])
            # merge strided transition scores
            self._dataset = self._dataset.apply(
                TransitionScores.merge, "transition_scores"
            )

            self.human_features: list[TransitionScores] = list(
                # filter for human written documents
                self._dataset.filter(
                    lambda doc: doc["document"]["type"] == "source", in_place=False
                )
                # filter out documents that are too short
                .filter(
                    lambda doc: len(doc["transition_scores"])
                    >= self.feature_selection.required_size()
                )
                .map(
                    lambda doc: {
                        "features": self.convert_to_features(doc["transition_scores"]),
                        "labels": 0,
                    }
                )
            )
            self.synth_features: list[TransitionScores] = list(
                # filter for AI generated documents
                self._dataset.filter(lambda doc: doc["document"]["type"] != "source")
                # filter out documents that are too short
                .filter(
                    lambda doc: len(doc["transition_scores"])
                    >= self.feature_selection.required_size()
                )
                .map(
                    lambda doc: {
                        "features": self.convert_to_features(doc["transition_scores"]),
                        "labels": 1,
                    }
                )
            )

        self.eval_data = []
        self.train_data = []

        human_eval_size = int(len(self.human_features) // (1 / self.eval_split_size))
        _train_data, _eval_data = torch.utils.data.random_split(
            self.human_features,
            [len(self.human_features) - human_eval_size, human_eval_size],
        )
        self.train_data.extend(_train_data)
        self.eval_data.extend(_eval_data)

        synth_eval_size = int(len(self.synth_features) // (1 / self.eval_split_size))
        _train_data, _eval_data = torch.utils.data.random_split(
            self.synth_features,
            [len(self.synth_features) - synth_eval_size, synth_eval_size],
        )
        self.train_data.extend(_train_data)
        self.eval_data.extend(_eval_data)

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
