from typing import Callable, Generator, Iterable, Iterator, TypedDict

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset as TorchDataset

from luminar.features import (
    FeatureExtractor,
    Likelihood,
    OneDimFeatures,
    SliceFirst,
    Slicer,
    TwoDimFeatures,
)
from luminar.mongo import MongoPipelineDataset, PrismaiDocument
from prismai_features.data import FeatureValues


def flatten[T](iterables: Iterable[Iterable[T]]) -> Generator[T, None, None]:
    yield from (item for iterable in iterables for item in iterable)


def default_label_fn(sample: dict) -> int:
    return 0 if sample["type"] == "source" else 1


class SampleDict(TypedDict):
    features: FeatureValues
    label: int


class FeatureDict(TypedDict):
    features: torch.Tensor
    label: int


class FeatureDataset(TorchDataset):
    def __init__(
        self,
        data: Iterable[FeatureDict],
    ):
        self._data: list[FeatureDict] = list(data)

    @classmethod
    def from_samples(
        cls,
        data: Iterable[SampleDict],
        slicer: Slicer,
        featurizer: FeatureExtractor,
        num_samples: int = 1,
    ):
        """
        Create a dataset for CNN training.
        Uses the given `slicer` to extract feature (potentially multiple) slices from each sample.
        Then applies the `featurizer` to each LLM feature slice to generate CNN input features for training.

        Args:
            data (Iterable[SampleDict]): An iterable of samples with LLM features and labels.
            slicer (Slicer): A `Slicer` instance to extract slices from the LLM features.
            featurizer (FeatureExtractor): A `FeatureExtractor` instance to extract CNN input features from the sliced LLM features.
            num_samples (int, optional): _description_. Defaults to 1.
        """
        return cls(
            [
                {
                    "features": featurizer.featurize(sample["features"], slice),
                    "labels": sample["label"],
                }
                for sample in data
                for slice in slicer.sample(len(sample["features"]), num_samples)
            ]
        )

    @classmethod
    def from_prismai(
        cls,
        dataset: Iterable[PrismaiDocument],
        slicer: Slicer,
        featurizer: FeatureExtractor,
        num_samples: int = 1,
        label_fn: Callable[[dict], int] = default_label_fn,
    ):
        """Create a dataset with features for CNN training from a list of documents.
        Each document is expected to have a list of samples.
        Adapt `label_fn` to infer the label from a sample.
        `label_fn` expects a dictionary with an entry "type" that is used to infer the label.

        Args:
            dataset (Iterable[PrismaiDocument]): An iterable over a list of documents.
            label_fn (Callable[[dict], int], optional): Function that infers the label from a sample. Defaults to default_label_fn.

        Returns:
            FeatureDataset: A dataset with features and labels.
        """
        return cls.from_samples(
            (
                SampleDict(
                    features=FeatureValues(**sample["features"]),
                    label=label_fn(sample),
                )
                for document in dataset
                for sample in document["samples"]
            ),
            slicer=slicer,
            featurizer=featurizer,
            num_samples=num_samples,
        )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index) -> FeatureDict:
        return self._data[index]

    def __iter__(self) -> Iterator[FeatureDict]:
        return iter(self._data)


def n_way_split(
    dataset: TorchDataset,
    *sizes: float,
    infer_first: bool = False,
) -> list[Subset]:
    """
    Split a dataset into n subsets with the given sizes.

    Args:
        dataset (TorchDataset): The dataset to split.
        sizes (float): The relative sizes of the subsets.
        infer_first (bool): Whether to infer the size of the first subset.
            If True, `len(sizes) + 1` Subsets will be returned.

    Returns:
        list[Subset]: The resulting subsets.
    """
    if infer_first:
        sizes = 1.0 - sum(sizes), *sizes

    return torch.utils.data.random_split(dataset, sizes)


class DocumentClassificationDataModule(LightningDataModule):
    def __init__(
        self,
        datasets: list[MongoPipelineDataset],
        feature_dim: OneDimFeatures | TwoDimFeatures = OneDimFeatures(256),
        slicer: Slicer | None = None,
        featurizer: FeatureExtractor | None = None,
        num_samples: int = 1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_split_size: float = 0.1,
        test_split_size: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        if not all(ds.use_cache for ds in datasets):
            raise ValueError("Dataset must use cache for this data module.")

        self.datasets = datasets
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
                "feature_dim": tuple(feature_dim),
                "slicer": str(self.slicer),
                "featurizer": str(self.featurizer),
            }
        )

        self._finished_setup = False

    def prepare_data(self):
        if not self._finished_setup:
            for dataset in self.datasets:
                if not dataset.get_cache_file().exists():
                    dataset.load()
                    dataset._data = None  # type: ignore

    def setup(self, stage=None):
        if not self._finished_setup:
            splits: list[list[Subset]] = []
            for dataset in self.datasets:
                splits.append(
                    n_way_split(
                        dataset,
                        self.hparams.eval_split_size,  # type: ignore
                        self.hparams.test_split_size,  # type: ignore
                        infer_first=True,
                    )
                )

            train_data, eval_data, test_data = zip(*splits)

            self.train_data = FeatureDataset.from_prismai(
                flatten(train_data), self.slicer, self.featurizer
            )
            self.eval_data = FeatureDataset.from_prismai(
                flatten(eval_data), self.slicer, self.featurizer
            )
            self.test_data = FeatureDataset.from_prismai(
                flatten(test_data), self.slicer, self.featurizer
            )

        self._finished_setup = True

    def train_dataloader(self):
        return PaddingDataloader(
            self.train_data,
            feature_dim=self.hparams.feature_dim,  # type: ignore
            batch_size=self.hparams.train_batch_size,  # type: ignore
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return PaddingDataloader(
            self.eval_data,
            feature_dim=self.hparams.feature_dim,  # type: ignore
            batch_size=self.hparams.eval_batch_size,  # type: ignore
            pin_memory=True,
        )

    def test_dataloader(self):
        return PaddingDataloader(
            self.test_data,
            feature_dim=self.hparams.feature_dim,  # type: ignore
            batch_size=self.hparams.eval_batch_size,  # type: ignore
            pin_memory=True,
        )


class PaddingDataloader(DataLoader):
    def __init__(self, *args, feature_dim: tuple[int, ...], **kwargs):
        kwargs["collate_fn"] = self._collate_fn
        super().__init__(*args, **kwargs)
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
                p2d = (0, d1 - s1)
                features = torch.nn.functional.pad(features, p2d, "constant", 0.0)
            case (_, s1, _), (d1, _) if s1 < d1:
                p2d = (0, 0, 0, d1 - s1, 0, 0)
                features = torch.nn.functional.pad(features, p2d, "constant", 0.0)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        labels = torch.tensor([x["labels"] for x in batch])

        return {"features": features, "labels": labels}
