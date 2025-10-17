from collections import defaultdict
from typing import Callable, Generator, Iterable, Sequence

import datasets
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


def get_matched_ids(
    dataset: Dataset,
    agent_set: set[str] | None,
    num_proc: int | None = None,
) -> set[str]:
    if agent_set is not None:
        ids_agent: set[str] = set(
            dataset.filter(
                lambda a: a in agent_set,
                input_columns=["agent"],
                num_proc=num_proc,
            )["id_source"]
        )
    else:
        ids_agent: set[str] = set(
            dataset.filter(
                lambda a: a != "human",
                input_columns=["agent"],
                num_proc=num_proc,
            )["id_source"]
        )

    ids_human: set[str] = set(
        dataset.filter(
            lambda a: a == "human",
            input_columns=["agent"],
            num_proc=num_proc,
        )["id_source"]
    )

    return ids_agent.intersection(ids_human)


def get_matched_datasets(
    dataset: Dataset,
    *agents: str | None,
    eval_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 42,
    num_proc: int | None = None,
) -> tuple[DatasetDict, Dataset]:
    datasets.disable_progress_bars()

    if not agents or agents == (None,):
        agent_set = None
    else:
        agent_set: set[str] = (
            set(agents) if isinstance(agents[0], str) else set(agents[0])
        )
    ids_matched_set = get_matched_ids(dataset, agent_set, num_proc=num_proc)
    agent_set.add("human")

    ids_matched = np.array(list(ids_matched_set), dtype=str)
    ids_matched.sort()
    np.random.seed(seed)
    np.random.shuffle(ids_matched)

    eval_offset = int(len(ids_matched) * (1 - eval_split - test_split))
    test_offset = int(len(ids_matched) * (1 - test_split))
    ids_train, ids_eval, ids_test = map(
        set, np.array_split(ids_matched, [eval_offset, test_offset])
    )

    if agents is not None:
        ds_agent = dataset.filter(
            lambda agent: agent in agent_set,
            input_columns=["agent"],
            num_proc=num_proc,
        )
    else:
        ds_agent = dataset  # use all agents

    dataset_matched = DatasetDict(
        {
            "train": ds_agent.filter(
                lambda _id: _id in ids_train,
                input_columns=["id_source"],
                num_proc=num_proc,
            ).shuffle(seed=seed),
            "eval": ds_agent.filter(
                lambda _id: _id in ids_eval,
                input_columns=["id_source"],
                num_proc=num_proc,
            ),
            "test": ds_agent.filter(
                lambda _id: _id in ids_test,
                input_columns=["id_source"],
                num_proc=num_proc,
            ),
        }
    )
    dataset_unmatched = dataset.filter(
        lambda _id: _id not in ids_matched_set,
        input_columns=["id_source"],
        num_proc=num_proc,
    )

    datasets.enable_progress_bars()
    return dataset_matched, dataset_unmatched


def get_matched_cross_validation_datasets(
    dataset: Dataset,
    *agents: str,
    num_splits: int = 10,
    eval_splits: int = 1,
    test_splits: int = 2,
    seed: int = 42,
    num_proc: int | None = None,
) -> tuple[list[DatasetDictTrainEvalTest], DatasetUnmatched]:
    assert eval_splits > 0 < test_splits, (
        "eval_splits & test_splits must be greater than 0"
    )
    assert eval_splits + test_splits < num_splits, (
        "eval_splits + test_splits must be less than num_splits"
    )

    agent_set: set[str] = set(agents) if isinstance(agents[0], str) else set(agents[0])
    ids_matched_set = get_matched_ids(dataset, agent_set)
    agent_set.add("human")

    ds_agent = dataset.filter(
        lambda agent: agent in agent_set,
        input_columns=["agent"],
        num_proc=num_proc,
    )

    ids_matched = np.array(list(ids_matched_set), dtype=str)
    ids_matched.sort()
    np.random.seed(seed)
    np.random.shuffle(ids_matched)

    ids_matched_splits: list[NDArray] = np.array_split(ids_matched, num_splits)

    dataset_splits = []
    for _ in range(num_splits):
        ids_test = set(flatten(ids_matched_splits[:test_splits]))
        ids_eval = set(
            flatten(ids_matched_splits[test_splits : test_splits + eval_splits])
        )
        ids_train = set(flatten(ids_matched_splits[test_splits + eval_splits :]))

        dataset_splits.append(
            DatasetDict(
                {
                    "train": ds_agent.filter(
                        lambda _id: _id in ids_train,
                        input_columns=["id_source"],
                    ).shuffle(seed=seed),
                    "eval": ds_agent.filter(
                        lambda _id: _id in ids_eval,
                        input_columns=["id_source"],
                    ),
                    "test": ds_agent.filter(
                        lambda _id: _id in ids_test,
                        input_columns=["id_source"],
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


def transpose_batch[K, V](batch: Sequence[dict[K, V]]) -> dict[K, list[V]]:
    """Transpose a batch of items &mdash; sequence of dictionaries &mdash; into a single dictionary of lists.
    All items **must** have the same keys!

    Returns:
        dict[K, list[V]]: The transposed batch as a dictionary.

    Raises:
        ValueError: If the items in the batch do not all have the same keys.
    """
    result = defaultdict(list)
    for item in batch:
        if result.keys() and result.keys() != item.keys():
            raise ValueError(
                f"Expected all items to have the same keys {result.keys()}, but got an item with keys {item.keys()}!"
            )

        for k, v in item.items():
            result[k].append(v)

    return dict(result)


def max_times_len(ll: list[int]) -> int:
    return max(ll) * len(ll)


def batched_dynamic[K, V](
    dataset: Iterable[dict[K, V]],
    max_effective_length: int,
    key: K | Callable[[dict[K, V]], int],
    len_fn: Callable[[list[int]], int] = max_times_len,
    max_batch_size: int | None = None,
) -> Generator[tuple[dict[K, V]], None, None]:
    """Create a Generator of batches that conform to an upper bound on the maximum effective length of the contained items.

    For the intended purpose &mdash; obtaining *padded* batches of input sequences that are smaller than the `max_effective_length` &mdash; pass the data *sorted by length* for best results.

    Args:
        dataset (Sequence[dict[K, V]]): A dataset of input sequences as a sequence of items (=dicts). Each item **must** have the same keys (not verified).
        max_effective_length (int): The total maximum effective length of a batch.
        key_fn (str | Callable[[dict[K, V]], int]): The key to retrieve the length of one item by. Can be a key in the item's dictionary or a callable that accepts an item as input.
        len_fn (Callable[[list[int]], int], optional): A function that calculates the effective length of a batch from their individual lengths, given as a list. Defaults to `max_times_len = lambda lengths: max(lengths) * len(lengths)`.
        max_batch_size (int | None, optional): If given, restrict the maximum number of items in a batch to this upper bound. Defaults to None.

    Yields:
        items (tuple[dict[K, V]]): Yields tuples of items that form a batch.
    """
    if callable(key):
        key_fn = key  # type: ignore
    else:

        def default_key_fn(item: dict[K, V]) -> int:
            return item[key]  # type: ignore

        key_fn = default_key_fn

    batch = []
    batch_lengths = []
    for item in dataset:
        # If the extended batch would exceed the max_effective_length, yield the current batch and start a new one
        item_len = key_fn(item)
        extended_length = len_fn(batch_lengths + [item_len])
        if (
            batch
            and extended_length > max_effective_length
            or max_batch_size
            and len(batch_lengths) + 1 > max_batch_size
        ):
            yield tuple(batch)

            batch.clear()
            batch_lengths.clear()

        batch.append(item)
        batch_lengths.append(item_len)

    if batch:
        yield tuple(batch)
