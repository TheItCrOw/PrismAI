import numpy as np
import torch
from attr import dataclass
from datasets import Dataset, DatasetDict
from numpy.typing import NDArray


def get_matched_datasets(
    dataset: Dataset,
    *agents: str,
    test_split: float = 0.2,
    eval_split: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    agent_set: set[str] = set(agents) if isinstance(agents[0], str) else set(agents[0])
    ids_agent: set[str] = set(
        dataset.filter(lambda a: a in agent_set, input_columns=["agent"])["id_source"]
    )
    ids_human: set[str] = set(
        dataset.filter(lambda a: a == "human", input_columns=["agent"])["id_source"]
    )
    ids_matched: NDArray = np.array(list(ids_agent.intersection(ids_human)), dtype=str)

    indices = np.arange(len(ids_matched))

    np.random.seed(seed)
    np.random.shuffle(indices)

    eval_offset = int(len(ids_matched) * (1 - eval_split - test_split))
    test_offset = int(len(ids_matched) * (1 - test_split))
    ids_train, ids_eval, ids_test = (
        set(ids_matched[indices[:eval_offset]]),
        set(ids_matched[indices[eval_offset:test_offset]]),
        set(ids_matched[indices[test_offset:]]),
    )

    agent_set.add("human")
    return DatasetDict(
        {
            "train": dataset.filter(
                lambda agent, _id: agent in agent_set and _id in ids_train,
                input_columns=["agent", "id_source"],
            ),
            "eval": dataset.filter(
                lambda agent, _id: agent in agent_set and _id in ids_eval,
                input_columns=["agent", "id_source"],
            ),
            "test": dataset.filter(
                lambda agent, _id: agent in agent_set and _id in ids_test,
                input_columns=["agent", "id_source"],
            ),
            "unmatched": dataset.filter(
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
