from hashlib import sha256
from pathlib import Path
from typing import Generator, Iterable, Self

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset as TorchDataset

from luminar.features import (
    AnyDimFeatures,
    FeatureExtractor,
    Likelihood,
    OneDimFeatures,
    SliceFirst,
    Slicer,
    TwoDimFeatures,
)
from luminar.mongo import MongoDataset
from transition_scores.data import TransitionScores


def flatten[T](iterables: Iterable[Iterable[T]]) -> Generator[T, None, None]:
    yield from (item for iterable in iterables for item in iterable)


class FeatureDataset(TorchDataset):
    def __init__(
        self,
        data: Iterable[dict],
        slicer: Slicer,
        featurizer: FeatureExtractor,
        num_samples: int = None,
        label_field: str = "type",
        label_zero: str = "source",
    ):
        self.data = list(
            flatten(
                [
                    {
                        "features": features,
                        "labels": 0 if sample[label_field] == label_zero else 1,
                    }
                    for sample in doc["features"]
                    for features in self._featurize(
                        TransitionScores(**sample["transition_scores"]),
                        slicer,
                        featurizer,
                        num_samples=num_samples or 1,
                    )
                ]
                for doc in data
            )
        )

    @staticmethod
    def _featurize(
        ts: TransitionScores,
        slicer: Slicer,
        featurizer: FeatureExtractor,
        num_samples: int,
    ) -> Iterable[dict[str, torch.Tensor]]:
        slices = slicer.sample(len(ts), num_samples)
        for s in slices:
            yield featurizer.featurize(ts, s)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        return self.data[index]

    def __iter__(self) -> Generator[dict, Self, None]:
        return iter(self.data)


def n_way_split(
    dataset: TorchDataset,
    *sizes: float,
    infer_first: bool = False,
) -> tuple[Subset, ...]:
    """
    Split a dataset into n subsets with the given sizes.

    Args:
        dataset (TorchDataset): The dataset to split.
        sizes (float): The relative sizes of the subsets.
        infer_first (bool): Whether to infer the size of the first subset.
            If True, `len(sizes) + 1` Subsets will be returned.

    Returns:
        tuple[Subset, ...]: The resulting subsets.
    """
    total_length = len(dataset)
    lengths = [int(total_length * size) for size in sizes]

    if infer_first:
        lengths.insert(0, total_length - sum(lengths))

    return torch.utils.data.random_split(dataset, lengths)


class DocumentClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: MongoDataset,
        feature_dim: OneDimFeatures | TwoDimFeatures = OneDimFeatures(256),
        slicer: Slicer = None,
        featurizer: FeatureExtractor = None,
        num_samples: int = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_split_size: float = 0.1,
        test_split_size: float = 0.2,
        load_from_cache: bool = True,
        **kwargs,
    ):
        super().__init__()
        if not dataset.use_cache:
            raise ValueError("Dataset must use cache for this data module.")

        self.dataset = dataset
        self.slicer = slicer or SliceFirst(feature_dim[0])
        self.featurizer = featurizer or Likelihood()

        match feature_dim:
            case (_,):
                feature_dim = OneDimFeatures(*feature_dim)
            case (_, _):
                feature_dim = TwoDimFeatures(*feature_dim)
            case _:
                raise ValueError(f"Invalid feature_dim {feature_dim}")

        self.save_hyperparameters(
            {
                "num_samples": num_samples,
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "eval_split_size": eval_split_size,
                "test_split_size": test_split_size,
                "feature_dim": feature_dim,
                "slicer": str(self.slicer),
                "featurizer": str(self.featurizer),
            }
        )

        self._load_from_cache
        self._finished_setup = False

    def prepare_data(self):
        if not self._finished_setup:
            if not self.dataset.get_cache_file().exists():
                self.dataset.load()
                self.dataset._data = None

    def setup(self, stage=None):
        if not self._finished_setup:
            _hash = sha256()
            _hash.update(f"num_samples={self.hparams['num_samples']}".encode())
            _hash.update(f"feature_dim={self.hparams['feature_dim']}".encode())
            _hash.update(f"slicer={self.hparams['slicer']}".encode())
            _hash.update(f"featurizer={self.hparams['featurizer']}".encode())
            _hash = _hash.hexdigest()

            dataset_cache_file = self.dataset.get_cache_file()
            cache_file = dataset_cache_file.with_suffix("") / f"{_hash}.pt"

            if self._load_from_cache:
                if cache_file.exists():
                    self.train_data, self.eval_data, self.test_data = torch.load(
                        cache_file
                    )
                    self._finished_setup = True
                    return

            if not self.eval_datasets and not self.test_datasets:
                splits: list[tuple[Subset, Subset, Subset]] = []
                for dataset in self.train_datasets:
                    splits.append(
                        n_way_split(
                            dataset,
                            self.hparams.eval_split_size,
                            self.hparams.test_split_size,
                            infer_first=True,
                        )
                    )

                train_data, eval_data, test_data = zip(*splits)
            elif not self.eval_datasets:
                splits: list[tuple[Subset, Subset]] = []
                for dataset in self.train_datasets:
                    splits.append(
                        n_way_split(
                            dataset,
                            self.hparams.eval_split_size,
                            infer_first=True,
                        )
                    )

                train_data, eval_data = zip(*splits)
                test_data = self.test_datasets
            elif not self.test_datasets:
                train_data = self.train_datasets
                eval_data = self.eval_datasets
                test_data = []
            else:
                train_data = self.train_datasets
                eval_data = self.eval_datasets
                test_data = self.test_datasets

            self.train_data = FeatureDataset(
                flatten(train_data), self.slicer, self.featurizer
            )
            self.eval_data = FeatureDataset(
                flatten(eval_data), self.slicer, self.featurizer
            )
            self.test_data = FeatureDataset(
                flatten(test_data), self.slicer, self.featurizer
            )

            if self._load_from_cache:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    (self.train_data, self.eval_data, self.test_data),
                    cache_file,
                )

        self._finished_setup = True

    def write_to_file(self, file: Path):
        with file.open("wb") as fp:
            torch.save(
                {
                    "hparams": self.hparams,
                    "slicer": self.slicer,
                    "featurizer": self.featurizer,
                    "train_data": self.train_data,
                    "eval_data": self.eval_data,
                    "test_data": self.test_data,
                },
                fp,
            )

    @classmethod
    def load_from_file(cls, file: Path):
        with file.open(mode="rb") as fp:
            data = torch.load(fp)
        kwargs = data["hparams"] | {
            "slicer": data["slicer"],
            "featurizer": data["featurizer"],
        }
        self = cls([], **kwargs)
        self.train_data = data["train_data"]
        self.eval_data = data["eval_data"]
        self.test_data = data["test_data"]
        self._finished_setup = True
        return self

    def train_dataloader(self):
        return PaddingDataloader(
            self.train_data,
            feature_dim=self.hparams.feature_dim,
            batch_size=self.hparams.train_batch_size,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return PaddingDataloader(
            self.eval_data,
            feature_dim=self.hparams.feature_dim,
            batch_size=self.hparams.eval_batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return PaddingDataloader(
            self.test_data,
            feature_dim=self.hparams.feature_dim,
            batch_size=self.hparams.eval_batch_size,
            pin_memory=True,
        )


class PaddingDataloader(DataLoader):
    def __init__(self, *args, feature_dim: tuple[int, ...], **kwargs):
        super().__init__(*args, collate_fn=self._collate_fn, **kwargs)
        self.feature_dim = feature_dim

    def _collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        features = torch.nn.utils.rnn.pad_sequence(
            [x["features"] for x in batch], batch_first=True
        )

        # In case we get a batch of sequences, that are all too short,
        # we need to pad them to the correct length as given by the feature_dim.
        # - First dimension is the batch size.
        # - Second dimension is the sequence length.
        # - Third dimension is the feature dimension, if 2D features are used.
        match features.shape, self.feature_dim:
            case (_, s1), (d1,) if s1 < d1:
                p2d = (0, 0, 0, d1 - s1)
                features = torch.nn.functional.pad(features, p2d, "constant", 0.0)
            case (_, s1, _), (d1, _) if s1 < d1:
                p2d = (0, 0, 0, d1 - s1, 0, 0)
                features = torch.nn.functional.pad(features, p2d, "constant", 0.0)
        labels = torch.tensor([x["labels"] for x in batch])

        return {"features": features, "labels": labels}
