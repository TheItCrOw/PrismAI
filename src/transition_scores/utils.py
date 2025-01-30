import enum
import gc
import os
from typing import Any, Final, Generator, Iterable

import numpy as np
import torch
from transformers import AutoConfig


class ModelIntializationError(Exception):
    pass


class DataClassMappingMixin:
    def keys(self):
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        if hasattr(self, key):
            value = getattr(self, key)
            if isinstance(value, DataClassMappingMixin):
                return dict(**value)
            return value
        raise KeyError(f"Key {key} not found in {type(self).__name__}")


def transpose_dict_of_lists[K, V](
    dd: dict[K, list[V]], iter: bool = False
) -> list[dict[K, V]] | Generator[dict[K, V], None, None]:
    """Transpose a dictionary of lists into a list of dictionaries.

    Args:
        dd (dict[K, list[V]]): A dictionary of lists to transpose.
        iter (bool, optional): If true, will yield transposed dictionaries
            instead of returning a list. Defaults to False.

    Returns:
        list[dict[K, V]]

    Yields:
        dict[K, V]
    """
    _gen = (dict(zip(dd, col)) for col in zip(*dd.values()))
    if iter:
        yield from _gen
    else:
        return list(_gen)


def chunks_to_text(chunks: list[str]) -> str:
    return " ".join(chunk.strip() for chunk in chunks)


def infer_max_length(model_name_or_path: str):
    config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    if hasattr(config, "n_positions"):
        return config.n_positions
    raise ValueError(f"Could not infer max length from {model_name_or_path}")


def flatten[T](nested: Iterable[Iterable[T]]) -> Generator[T, None, None]:
    yield from (item for inner in nested for item in inner)


def sort_by_column(
    dataset: list[dict[str, Any | list]], sort_by: str, fields_to_sort: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Sort the given `fields_to_sort` by the values provided in the given `sort_by` column.

    Args:
        dataset (list[dict[str, Any  |  list]]): The dataset to sort.
        sort_by (str): The column to sort by containing lists of sortable values.
        fields_to_sort (tuple[str, ...]): The fields to sort.
            All fields must be lists of the same length as the column.

    Returns:
        list[dict[str, Any]]: The sorted dataset.
    """
    # lists & dicts are mutable, so we can just manipulate the values in-place
    for row in dataset:
        order = np.argsort(row[sort_by])
        for field in fields_to_sort:
            if len(row[field]) != len(row[sort_by]):
                raise ValueError(
                    f"Field '{field}' has a different length than the column '{sort_by}': {len(row[field])} != {len(row[sort_by])}"
                )
            row[field] = [row[field][i] for i in order]
    return dataset


def group_by_column(
    dataset: list[dict[str, Any]],
    key_column: str = "_id",
    deduplicate: tuple[str, ...] | None = ("_id",),
    aggregate: tuple[str, ...] | None = None,
    into: str = "grouped",
    pop_key_column: bool = False,
) -> list[dict[str, Any]]:
    """Group a dataset by a column and move other columns into a list.

    Examples:
        >>> dataset = [
        ...     {"foo": 1, "bar": "baz", "values": [1,2,3]},
        ...     {"foo": 1, "bar": "baz", "values": [4,5,6]},
        ...     {"foo": 2, "bar": "qux", "values": [7,8,9]},
        ... ]
        >>> group_by_column(dataset, "foo", ("foo", "bar",), ("values",))
        [{'foo': 1, 'bar': 'baz', 'values': [[1, 2, 3], [4, 5, 6]]}, {'foo': 2, 'bar': 'qux', 'values': [[7, 8, 9]]}]

    Args:
        dataset (list[dict[str, Any]]): The dataset to group.
        column (str): The column to group by.
        deduplicate (tuple[str, ...]): Columns to deduplicate.
            These columns will be moved into the parent dict and not aggregated into a list.
            Duplicate values will be overwritten.
        aggregate (tuple[str, ...]): Columns to map.
            These columns will be moved into the parent dict and aggregated into a list.
        into (str | None): The target column to move all remaining values into
            that are not covered by `deduplicate` or `aggregate`.

    Returns:
        dict[str, list[dict[str, Any]]]: The grouped dataset.
    """
    grouped = dict()
    for source in dataset:
        key = source.pop(key_column) if pop_key_column else source[key_column]
        target = grouped.setdefault(key, dict())

        if deduplicate:
            for k in deduplicate:
                if k in source:
                    target[k] = source.pop(k)

        if aggregate:
            for k in aggregate:
                if k in source:
                    target.setdefault(k, []).append(source.pop(k))

        if source:
            target.setdefault(into, []).append(source)
    return list(grouped.values())


class PytorchGcLevel(enum.IntEnum):
    """
    Aggressiveness level for PyTorch garbage collection.
    NONE: Default Python garbage collection.
    DATASET: Garbage collection after each dataset.
    BATCH: Garbage collection after each batch.
    """

    NONE = OFF = FALSE = DEFAULT = 0
    DATASET = 1
    BATCH = 2


_env_pytorch_gc_level = os.environ.get("PYTORCH_GC_LEVEL", "NONE").strip().upper()
PYTORCH_GC_LEVEL: Final[PytorchGcLevel] = PytorchGcLevel(
    PytorchGcLevel(int(_env_pytorch_gc_level))
    if _env_pytorch_gc_level.isdigit()
    else PytorchGcLevel[_env_pytorch_gc_level]
)


def free_memory():
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
