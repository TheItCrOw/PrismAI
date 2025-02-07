import itertools
import pickle
from typing import Iterable

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from luminar.features import (
    AnyDimFeatures,
    FeatureExtractor,
    OneDimFeatures,
    Slicer,
)
from luminar.mongo import MongoDBAdapter
from simple_dataset.dataset import Dataset
from transition_scores.data import TransitionScores


class DocumentClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        mongodb_adapter: MongoDBAdapter,
        feature_dim: AnyDimFeatures | tuple[int, ...] = OneDimFeatures(256),
        slicer: Slicer | Slicer.Type | str = Slicer.Type.First,
        featurizer: FeatureExtractor
        | FeatureExtractor.Type
        | str = FeatureExtractor.Type.Likelihood,
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
                "slicer": str(slicer),
                "featurizer": str(featurizer),
            }
        )

        self.mongodb_adapter = mongodb_adapter
        match slicer:
            case instance if isinstance(instance, Slicer):
                self.slicer: Slicer = instance
            case value if isinstance(value, Slicer.Type):
                self.slicer: Slicer = value.into(feature_dim)
            case string if isinstance(string, str):
                self.slicer: Slicer = Slicer.Type[string].into(feature_dim)
            case _:
                raise ValueError(
                    f"Unsupported feature selection: {slicer}. "
                    f"Must be a {Slicer.__name__} instance one of {Slicer.Type.__members__.keys()}"
                )

        match featurizer:
            case instance if isinstance(instance, FeatureExtractor):
                self.featurizer: FeatureExtractor = instance
            case value if isinstance(value, FeatureExtractor.Type):
                self.featurizer: FeatureExtractor = value.into()
            case string if isinstance(string, str):
                self.featurizer: FeatureExtractor = FeatureExtractor.Type[string].into()
            case _:
                raise ValueError(
                    f"Unsupported feature type: {featurizer}. "
                    f"Must be a {FeatureExtractor.Type.__name__} instance one of {FeatureExtractor.Type.__members__.keys()}"
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
            case "fit" | "train" if getattr(self, "train_data", None) is not None:
                return
            case "validation" if getattr(self, "eval_data", None) is not None:
                return
            case "test" if getattr(self, "test_data", None) is not None:
                return
            case "predict":
                raise NotImplementedError(stage)
            case None:
                pass

        if getattr(self, "_dataset", None) is None:
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

            # flatten the nested features list
            self._dataset = self._dataset.flat_map(lambda doc: doc["features"])
            # merge strided transition scores
            self._dataset = self._dataset.apply(
                TransitionScores.merge, "transition_scores"
            )

            self._dataset: dict[str, list[TransitionScores]] = (
                self._dataset.group_documents_by(
                    lambda doc: doc["document"].get("type", "source"),
                    aggregate=("transition_scores",),
                    return_dict=True,
                )
            )

        if (
            getattr(self, "human_features", None) is None
            or getattr(self, "synth_features", None) is None
        ):
            keys = set(self._dataset.keys())
            if "source" not in keys:
                raise ValueError(f"Expected a 'source' document type, but got {keys}")
            if len(self._dataset.keys()) > 2:
                raise ValueError(f"Expected only two document types, but got {keys}")

            key = "source"
            keys.remove(key)
            self.human_features = (
                Dataset(self._dataset[key]["transition_scores"])
                .flat_map(self._featurize)
                .update(itertools.cycle([{"labels": 0}]))
            )

            (key,) = keys
            self.synth_features = (
                Dataset(self._dataset[key]["transition_scores"])
                .flat_map(self._featurize)
                .update(itertools.cycle([{"labels": 1}]))
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

    def _featurize(self, ts: TransitionScores) -> Iterable[dict[str, torch.Tensor]]:
        slices = self.slicer.slice(len(ts))
        features = self.featurizer.featurize(ts, slices)
        yield from ({"features": feature} for feature in features)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data,
            batch_size=self.eval_batch_size,
            collate_fn=self._collate_fn,
        )

    # def test_dataloader(self):
    #     if len(self.eval_splits) == 1:
    #         return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
    #     elif len(self.eval_splits) > 1:
    #         return [
    #             DataLoader(self.dataset[x], batch_size=self.eval_batch_size)
    #             for x in self.eval_splits
    #         ]

    def _collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        features = torch.nn.utils.rnn.pad_sequence(
            [x["features"] for x in batch], batch_first=True
        )
        features = torch.nn.functional.pad(
            features, (0, self.hparams.feature_dim[0] - features.size(1))
        )
        labels = torch.tensor([x["labels"] for x in batch])

        return {"features": features, "labels": labels}
