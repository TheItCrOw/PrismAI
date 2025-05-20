import torch
from attr import dataclass
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader


def get_matched_datasets(
    dataset: Dataset,
    agent: str,
    test_size=0.2,
    seed=42,
) -> DatasetDict:
    ids_human: set[str] = set(
        dataset.filter(lambda a: a == "human", input_columns=["agent"])["id_source"]
    )

    ds_agent: DatasetDict = (
        dataset.filter(lambda a: a == agent, input_columns=["agent"])
        .remove_columns(list(set(dataset.column_names).difference({"id_source"})))
        .train_test_split(test_size, seed=seed)
    )

    ids_train: set[str] = set(ds_agent["train"]["id_source"]).intersection(ids_human)  # type: ignore
    ids_eval: set[str] = set(ds_agent["test"]["id_source"]).intersection(ids_human)  # type: ignore
    ids_matched: set[str] = ids_train | ids_eval

    return DatasetDict(
        {
            "train": dataset.filter(
                lambda _id: _id in ids_train, input_columns=["id_source"]
            ),
            "test": dataset.filter(
                lambda _id: _id in ids_eval, input_columns=["id_source"]
            ),
            "test_unmatched": dataset.filter(
                lambda _id: _id not in ids_matched, input_columns=["id_source"]
            ),
        }
    )


def collate_pad_features(
    feature_dim: tuple[int, ...],
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    features = torch.nn.utils.rnn.pad_sequence(
        [item["features"] for item in batch], batch_first=True
    )

    # In case we get a batch of sequences, that are all too short,
    # we need to pad them to the correct length as given by the feature_dim.
    # - First dimension is the batch size.
    # - Second dimension is the sequence length.
    # - Third dimension is the feature dimension, if 2D features are used.
    match features.shape, feature_dim:
        case (_, s1), (d1,) if s1 < d1:
            p2d = (0, d1 - s1)
            features = torch.nn.functional.pad(features, p2d, "constant", 0.0)
        case (_, s1, _), (d1, _) if s1 < d1:
            p2d = (0, 0, 0, d1 - s1, 0, 0)
            features = torch.nn.functional.pad(features, p2d, "constant", 0.0)

    features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    labels = torch.tensor([item["labels"] for item in batch])

    return {"features": features, "labels": labels}


class PaddingDataloader(DataLoader):
    def __init__(self, *args, feature_dim: tuple[int, ...], **kwargs):
        kwargs["collate_fn"] = self._collate_fn
        super().__init__(*args, **kwargs)
        self.feature_dim = feature_dim

    def _collate_fn(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        return collate_pad_features(self.feature_dim, batch)


@dataclass
class PaddingDataCollator:
    feature_dim: tuple[int, ...]

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return collate_pad_features(self.feature_dim, batch)
