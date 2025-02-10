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
from luminar.mongo import MongoDatset
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
    ):
        self.data = list(
            flatten(
                [
                    {
                        "features": features,
                        "labels": 0 if sample["type"] == "source" else 1,
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
        *train_datasets: MongoDatset,
        eval_datasets: list[MongoDatset] = None,
        test_datasets: list[MongoDatset] = None,
        feature_dim: OneDimFeatures | TwoDimFeatures = OneDimFeatures(256),
        slicer: Slicer = None,
        featurizer: FeatureExtractor = None,
        num_samples: int = None,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_split_size: float = 0.1,
        test_split_size: float = 0.2,
        **kwargs,
    ):
        super().__init__()

        self.slicer = slicer or SliceFirst(feature_dim[0])
        self.featurizer = featurizer or Likelihood()

        self.save_hyperparameters(
            {
                "num_samples": num_samples,
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "eval_split_size": eval_split_size,
                "test_split_size": test_split_size,
                "feature_dim": tuple(feature_dim),
                "slicer": str(self.slicer),
                "featurizer": str(self.featurizer),
            }
        )

        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets
        self.test_datasets = test_datasets

        self.__finished_setup = False

    def prepare_data(self):
        for dataset in self.train_datasets:
            if dataset.use_cache:
                dataset.load()

        if self.eval_datasets:
            for dataset in self.eval_datasets:
                if dataset.use_cache:
                    dataset.load()

        if self.test_datasets:
            for dataset in self.test_datasets:
                if dataset.use_cache:
                    dataset.load()

        return super().prepare_data()

    def setup(self, _stage=None):
        if not self.__finished_setup:
            if not self.eval_datasets and not self.test_datasets:
                splits: list[tuple[Subset, Subset, Subset]] = []
                for dataset in self.train_datasets:
                    total_length = len(dataset)
                    size_val = int(total_length * self.hparams.eval_split_size)
                    size_test = int(total_length * self.hparams.test_split_size)
                    size_train = total_length - size_val - size_test
                    splits.append(
                        n_way_split(
                            dataset,
                            self.hparams.eval_split_size,
                            self.hparams.test_split_size,
                        )
                    )

                train_data, eval_data, test_data = zip(*splits)
            elif not self.eval_datasets:
                splits: list[tuple[Subset, Subset]] = []
                for dataset in self.train_datasets:
                    total_length = len(dataset)
                    size_val = int(total_length * self.hparams.eval_split_size)
                    size_train = total_length - size_val
                    splits.append(
                        torch.utils.data.random_split(dataset, [size_train, size_val])
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

            self.__finished_setup = True

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
        features = torch.nn.functional.pad(
            features, (0, self.feature_dim[0] - features.size(1))
        )
        labels = torch.tensor([x["labels"] for x in batch])

        return {"features": features, "labels": labels}
