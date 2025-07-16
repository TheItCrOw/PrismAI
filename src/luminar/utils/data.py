from typing import Callable, Generator, Iterable

import numpy as np
import torch
from attr import dataclass
from datasets import Dataset, DatasetDict
from numpy.typing import NDArray

type DatasetUnmatched = Dataset
type DatasetDictTrainEvalTest = DatasetDict


def flatten[T](outer: Iterable[Iterable[T]]) -> Generator[T, None, None]:
    for inner in outer:
        yield from inner


def get_matched_ids(dataset: Dataset, agent_set: set[str]) -> set[str]:
    ids_agent: set[str] = set(
        dataset.filter(lambda a: a in agent_set, input_columns=["agent"])["id_source"]
    )
    ids_human: set[str] = set(
        dataset.filter(lambda a: a == "human", input_columns=["agent"])["id_source"]
    )
    return ids_agent.intersection(ids_human)


def get_matched_datasets(
    dataset: Dataset,
    *agents: str,
    eval_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 42,
) -> tuple[DatasetDict, Dataset]:
    agent_set: set[str] = set(agents) if isinstance(agents[0], str) else set(agents[0])
    ids_matched = np.array(list(get_matched_ids(dataset, agent_set)), dtype=str)
    agent_set.add("human")

    ids_matched.sort()
    np.random.seed(seed)
    np.random.shuffle(ids_matched)

    eval_offset = int(len(ids_matched) * (1 - eval_split - test_split))
    test_offset = int(len(ids_matched) * (1 - test_split))
    ids_train, ids_eval, ids_test = map(
        set, np.array_split(ids_matched, [eval_offset, test_offset])
    )

    dataset_matched = DatasetDict(
        {
            "train": dataset.filter(
                lambda agent, _id: agent in agent_set and _id in ids_train,
                input_columns=["agent", "id_source"],
            ).shuffle(seed=seed),
            "eval": dataset.filter(
                lambda agent, _id: agent in agent_set and _id in ids_eval,
                input_columns=["agent", "id_source"],
            ),
            "test": dataset.filter(
                lambda agent, _id: agent in agent_set and _id in ids_test,
                input_columns=["agent", "id_source"],
            ),
        }
    )
    dataset_unmatched = dataset.filter(
        lambda _id: _id not in ids_matched, input_columns=["id_source"]
    )

    return dataset_matched, dataset_unmatched


def get_matched_cross_validation_datasets(
    dataset: Dataset,
    *agents: str,
    num_splits: int = 10,
    eval_splits: int = 1,
    test_splits: int = 2,
    seed: int = 42,
) -> tuple[list[DatasetDictTrainEvalTest], DatasetUnmatched]:
    assert eval_splits > 0 < test_splits, (
        "eval_splits & test_splits must be greater than 0"
    )
    assert eval_splits + test_splits < num_splits, (
        "eval_splits + test_splits must be less than num_splits"
    )

    agent_set: set[str] = set(agents) if isinstance(agents[0], str) else set(agents[0])
    ids_matched_set = get_matched_ids(dataset, agent_set)
    ids_matched = np.array(list(ids_matched_set), dtype=str)
    agent_set.add("human")

    ids_matched.sort()
    np.random.seed(seed)
    np.random.shuffle(ids_matched)

    ids_matched_splits: list[NDArray] = np.array_split(ids_matched, num_splits)

    dataset_splits = []
    for _ in range(num_splits):
        ids_eval = set(flatten(ids_matched_splits[:eval_splits]))
        ids_test = set(
            flatten(ids_matched_splits[eval_splits : eval_splits + test_splits])
        )
        ids_train = set(flatten(ids_matched_splits[eval_splits + test_splits :]))

        dataset_splits.append(
            DatasetDict(
                {
                    "train": dataset.filter(
                        lambda agent, _id: agent in agent_set and _id in ids_train,
                        input_columns=["agent", "id_source"],
                    ).shuffle(seed=seed),
                    "eval": dataset.filter(
                        lambda agent, _id: agent in agent_set and _id in ids_eval,
                        input_columns=["agent", "id_source"],
                    ),
                    "test": dataset.filter(
                        lambda agent, _id: agent in agent_set and _id in ids_test,
                        input_columns=["agent", "id_source"],
                    ),
                }
            )
        )

        # Rotate the splits for the next iteration
        ids_matched_splits.append(ids_matched_splits.pop(0))

    return dataset_splits, dataset.filter(
        lambda _id: _id not in ids_matched_set, input_columns=["id_source"]
    )


@dataclass
class PaddingDataCollator:
    feature_dim: tuple[int, ...]

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        features = torch.nn.utils.rnn.pad_sequence(
            [torch.zeros(self.feature_dim)] + [item["features"] for item in batch],
            batch_first=True,
        )[1:]
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        labels = torch.tensor([item["labels"] for item in batch])

        return {"features": features, "labels": labels}


@dataclass
class MaskingDataCollator:
    feature_dim: tuple[int, ...]

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        features = torch.nn.utils.rnn.pad_sequence(
            [torch.zeros(self.feature_dim)] + [item["features"] for item in batch],
            batch_first=True,
        )[1:]
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        attention_mask = torch.full_like(features, False, dtype=torch.bool)
        for i, item in enumerate(batch):
            attention_mask[i, : item["features"].shape[0]] = True

        labels = torch.tensor([item["labels"] for item in batch])

        return {"features": features, "labels": labels}


def get_pad_to_fixed_length_fn(feature_len: int) -> Callable[[NDArray], NDArray]:
    def pad_to_fixed_length(x: NDArray) -> NDArray:
        size = x.shape[0]
        if size < feature_len:
            pad_size = ((0, feature_len - size), (0, 0))
            try:
                return np.pad(x, pad_size, mode="constant")
            except Exception as e:
                raise ValueError(
                    f"Failed to pad features of shape {x.shape} with np.pad({pad_size})"
                ) from e

        else:
            return x[:feature_len]

    return pad_to_fixed_length
