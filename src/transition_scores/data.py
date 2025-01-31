from collections import UserList
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Callable,
    Concatenate,
    Iterable,
    Literal,
    NamedTuple,
    ParamSpec,
    Self,
)

import numpy as np
from bson import ObjectId

from transition_scores.utils import DataClassMappingMixin


class OutputProbabilities(NamedTuple):
    target_probs: list[float]
    top_k_indices: list[list[int]]
    top_k_probs: list[list[float]]


@dataclass
class TransitionScores(DataClassMappingMixin):
    target_ids: list[int] = field(default_factory=list)
    target_probs: list[float] = field(default_factory=list)
    top_k_indices: list[list[int]] = field(default_factory=list)
    top_k_probs: list[list[float]] = field(default_factory=list)

    def __iter__(self):
        yield from zip(
            self.target_ids,
            self.target_probs,
            self.top_k_indices,
            self.top_k_probs,
        )

    def __getitem__(self, key):
        if isinstance(key, (int, slice)):
            return TransitionScores(
                self.target_ids[key],
                self.target_probs[key],
                self.top_k_indices[key],
                self.top_k_probs[key],
            )
        return super().__getitem__(key)

    def append(
        self,
        target_id: int,
        target_probs: float,
        top_k_indices: list[int],
        top_k_probs: list[float],
    ):
        self.target_ids.append(target_id)
        self.target_probs.append(target_probs)
        self.top_k_indices.append(top_k_indices)
        self.top_k_probs.append(top_k_probs)

    def extend(
        self,
        target_ids: list[int],
        target_probs: list[float],
        top_k_indices: list[list[int]],
        top_k_probs: list[list[float]],
    ):
        self.target_ids.extend(target_ids)
        self.target_probs.extend(target_probs)
        self.top_k_indices.extend(top_k_indices)
        self.top_k_probs.extend(top_k_probs)

    @classmethod
    def merge(cls, others: list[Self]) -> Self:
        self = cls()

        for other in others:
            self.extend(
                other.target_ids,
                other.target_probs,
                other.top_k_indices,
                other.top_k_probs,
            )
        return self


class ModelMetadata(dict):
    @classmethod
    def new(
        cls,
        name: str,
        provider: Literal["onnx", "transformers"],
        variant: Literal["default", "8bit", "o1", "o2", "o3", "o4"] = "default",
        **metadata,
    ) -> Self:
        return cls(
            {
                "name": name,
                "provider": provider,
                "variant": variant,
                **metadata,
            }
        )

    @classmethod
    def from_tuple(cls, tup: tuple) -> Self:
        return cls.new(*tup)


class PreProcessorMetadata(dict):
    @classmethod
    def new(
        cls,
        type: Literal["text", "chunk", "chunk-in-context"],
        **metadata,
    ) -> Self:
        return cls(
            {
                "type": type,
                **metadata,
            }
        )

    @classmethod
    def from_tuple(cls, tup: tuple) -> Self:
        return cls.new(*tup)


class FeaturesDict(dict):
    @classmethod
    def new(
        cls,
        refs: dict,
        text_sha256: str,
        model: ModelMetadata | dict,
        pre_processor: PreProcessorMetadata | dict,
        transition_scores: list[TransitionScores],
        **metadata,
    ) -> Self:
        return cls(
            {
                "_id": ObjectId(),
                "refs": refs,
                "text_sha256": text_sha256,
                "model": model,
                "pre_processor": pre_processor,
                "transition_scores": transition_scores,
                "metadata": metadata,
            }
        )

    @classmethod
    def from_tuple(cls, tup: tuple) -> Self:
        return cls.new(*tup)

    def split(self) -> tuple[Self, Self]:
        _split = self.get("_split", str(self["_id"]))
        ts = self["transition_scores"]
        tsa, tsb = (
            ts[: len(ts) // 2],
            ts[len(ts) // 2 :],
        )
        ma, mb = {}, {}
        for key, value in self["metadata"].items():
            if isinstance(value, list):
                ma[key], mb[key] = (
                    value[: len(value) // 2],
                    value[len(value) // 2 :],
                )
            else:
                ma[key], mb[key] = value, value
        return (
            FeaturesDict(
                {
                    "_id": ObjectId(),
                    "_split": _split + ".0",
                    "refs": self["refs"],
                    "text_sha256": self["text_sha256"],
                    "model": self["model"],
                    "pre_processor": self["pre_processor"],
                    "transition_scores": tsa,
                    "metadata": ma,
                }
            ),
            FeaturesDict(
                {
                    "_id": ObjectId(),
                    "_split": _split + ".1",
                    "refs": self["refs"],
                    "text_sha256": self["text_sha256"],
                    "model": self["model"],
                    "pre_processor": self["pre_processor"],
                    "transition_scores": tsb,
                    "metadata": mb,
                }
            ),
        )


P = ParamSpec("P")


class Dataset[K, V](UserList[dict[K, V]]):
    def map(
        self,
        map_fn: Callable[[dict[K, V]], dict[K, V]],
        in_place: bool = True,
    ) -> Self:
        """
        Map the dataset using the provided map function.

        Args:
            map_fn (Callable): A function that takes a document and returns a document.
            in_place (bool): Apply the operation **in-place**, modifying the original dataset.
                Default: `True`.

        Returns:
            Dataset: The mapped dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"foo": 1, "bar": "baz"},
            ...     {"foo": 2, "bar": "qux"},
            ... ])
            >>> dataset.map(lambda x: {"abc": x.pop("foo") * 2} | x).data
            [{'abc': 2, 'bar': 'baz'}, {'abc': 4, 'bar': 'qux'}]
        """
        if in_place:
            self.data = list(map(map_fn, self.data))
            return self
        else:
            return Dataset(map(map_fn, self.data))

    def flat_map(
        self,
        map_fn: Callable[[dict[K, V]], Iterable[dict[K, V]]],
        in_place: bool = True,
    ) -> Self:
        """
        Flat-map the dataset using the provided map function.

        Args:
            map_fn (Callable): A function that takes a document and returns an iterable of documents.
            in_place (bool): Apply the operation **in-place**, modifying the original dataset.
                Default: `True`.

        Returns:
            Dataset: The flat-mapped dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"foo": 1, "bar": "baz"},
            ...     {"foo": 2, "bar": "qux"},
            ... ])
            >>> dataset.flat_map(lambda x: (x, x)).data
            [{'foo': 1, 'bar': 'baz'}, {'foo': 1, 'bar': 'baz'}, {'foo': 2, 'bar': 'qux'}, {'foo': 2, 'bar': 'qux'}]
        """
        if in_place:
            self.data = list(doc for document in self.data for doc in map_fn(document))
            return self
        else:
            return Dataset(doc for document in self.data for doc in map_fn(document))

    def flat_map_zip(
        self,
        map_fn: Callable[Concatenate[dict[K, V], P], Iterable[dict[K, V]]],
        *iterables: P.args,
        in_place: bool = True,
    ) -> Self:
        """
        Flat-map the dataset using the provided map function.

        Args:
            map_fn (Callable): A function that takes a document and returns an iterable of documents.
            in_place (bool): Apply the operation **in-place**, modifying the original dataset.
                Default: `True`.

        Returns:
            Dataset: The flat-mapped dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"foo": 1, "bar": "baz"},
            ...     {"foo": 2, "bar": "qux"},
            ... ])
            >>> dataset.flat_map_zip(lambda x, ys: (x|y for y in ys), [[{"baz": 123}, {"baz": 456}],[{"baz": 789}]]).data
            [{'foo': 1, 'bar': 'baz', 'baz': 123}, {'foo': 1, 'bar': 'baz', 'baz': 456}, {'foo': 2, 'bar': 'qux', 'baz': 789}]
        """
        if in_place:
            self.data = list(
                doc
                for (document, *args) in zip(self.data, *iterables)
                for doc in map_fn(document, *args)
            )
            return self
        else:
            return Dataset(
                doc
                for (document, *args) in zip(self.data, *iterables)
                for doc in map_fn(document, *args)
            )

    def update(
        self,
        values: Iterable[dict[K, V]],
    ) -> Self:
        """
        Update the dataset using the provided map function.

        Args:
            others (Iterable): An iterable of documents (or another dataset).

        Returns:
            Dataset: The updated dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"foo": 1, "bar": "baz", "values": [1,2,3]},
            ...     {"foo": 1, "bar": "baz", "values": [4,5,6]},
            ...     {"foo": 2, "bar": "qux", "values": [7,8,9]},
            ... ])
            >>> dataset.update([{"values": [1,2]}, {"values": [4,5]}, {"values": [7,8]}]).data
            [{'foo': 1, 'bar': 'baz', 'values': [1, 2]}, {'foo': 1, 'bar': 'baz', 'values': [4, 5]}, {'foo': 2, 'bar': 'qux', 'values': [7, 8]}]
        """
        for document, other in zip(self.data, values):
            document.update(other)
        return self

    def modify(
        self,
        map_fn: Callable[[dict[K, V]], None],
    ) -> Self:
        """
        Modify the dataset using the provided map function.

        Args:
            map_fn (Callable): A function that takes a document and modifies it in-place.

        Returns:
            Dataset: The modified dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"foo": 1, "bar": "baz", "values": [1,2,3]},
            ...     {"foo": 1, "bar": "baz", "values": [4,5,6]},
            ...     {"foo": 2, "bar": "qux", "values": [7,8,9]},
            ... ])
            >>> dataset.modify(lambda x: x.update({"values": [1,2,3]})).data
            [{'foo': 1, 'bar': 'baz', 'values': [1, 2, 3]}, {'foo': 1, 'bar': 'baz', 'values': [1, 2, 3]}, {'foo': 2, 'bar': 'qux', 'values': [1, 2, 3]}]
        """
        for document in self.data:
            map_fn(document)
        return self

    def modify_zip(
        self,
        map_fn: Callable[Concatenate[dict[K, V], P], None],
        *iterables: P.args,
    ) -> Self:
        """
        Modify the dataset using the provided map function, zipping the dataset with other datasets.

        Args:
            map_fn (Callable): A function that takes a document the same number of positional arguments as passed to this method in `args`, modifying the document in-place.
            *iterables: Any number of iterables to zip with the dataset.

        Returns:
            Dataset: The modified dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"foo": 1, "bar": "baz", "values": [1,2,3]},
            ...     {"foo": 1, "bar": "baz", "values": [4,5,6]},
            ...     {"foo": 2, "bar": "qux", "values": [7,8,9]},
            ... ])
            >>> dataset.modify_zip(lambda x, y: x.update(y), [{"foo": 3}, {"foo": 4}, {"foo": 5}]).data
            [{'foo': 3, 'bar': 'baz', 'values': [1, 2, 3]}, {'foo': 4, 'bar': 'baz', 'values': [4, 5, 6]}, {'foo': 5, 'bar': 'qux', 'values': [7, 8, 9]}]
        """
        for document, *args in zip(self.data, *iterables):
            map_fn(document, *args)
        return self

    def filter(
        self,
        filter_fn: Callable[[dict[K, V]], bool],
        in_place: bool = True,
    ):
        """
        Filter the dataset using the provided filter function.

        Args:
            filter_fn (Callable): A function that takes a document and returns a boolean.
            in_place (bool): Apply the operation **in-place**, modifying the original dataset.
                Default: `True`.

        Returns:
            Dataset: The filtered dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"foo": 1, "bar": "baz", "values": [1,2,3]},
            ...     {"foo": 1, "bar": "baz", "values": [4,5,6]},
            ...     {"foo": 2, "bar": "qux", "values": [7,8,9]},
            ... ])
            >>> dataset.filter(lambda x: x["foo"] == 1).data
            [{'foo': 1, 'bar': 'baz', 'values': [1, 2, 3]}, {'foo': 1, 'bar': 'baz', 'values': [4, 5, 6]}]
        """
        if in_place:
            self.data = list(filter(filter_fn, self.data))
            return self
        else:
            return Dataset(filter(filter_fn, self.data))

    def remove_columns(
        self,
        *columns: K,
        in_place: bool = True,
    ) -> Self:
        """
        Remove the specified columns from the dataset.

        Args:
            *columns (K): A sequence of column names to remove.
            in_place (bool): Apply the operation **in-place**, modifying the original dataset.
                Default: `True`.

        Returns:
            Dataset: The dataset with the specified columns removed.

        Examples:
            >>> dataset = Dataset([
            ...     {"foo": 1, "bar": "baz", "values": [1,2,3]},
            ...     {"foo": 1, "bar": "baz", "values": [4,5,6]},
            ...     {"foo": 2, "bar": "qux", "values": [7,8,9]},
            ... ])
            >>> dataset.remove_columns("foo", "bar").data
            [{'values': [1, 2, 3]}, {'values': [4, 5, 6]}, {'values': [7, 8, 9]}]
        """
        columns = set(columns)
        if in_place:
            for document in self.data:
                for column in columns:
                    document.pop(column, None)
            return self
        else:
            return Dataset(
                {key: value for key, value in document.items() if key not in columns}
                for document in self.data
            )

    def group_by_column(
        self,
        by: K,
        deduplicate: tuple[K, ...] | None = None,
        aggregate: tuple[K, ...] | None = None,
        remainder_into: K | None = None,
        in_place: bool = True,
    ) -> Self:
        """Group a dataset by a column and move other columns into a list.

        Args:
            by (K): The column to group by.
                If not present in``deduplicate` or `aggregate`, it will be removed.
            deduplicate (tuple[K, ...]): Columns to deduplicate.
                These columns will be moved into the parent dict and not aggregated into a list.
                Duplicate values will be overwritten in the order they appear in the dataset.
            aggregate (tuple[K, ...]): Columns to aggregate into lists in the grouped dataset.
                Note: values already present in `deduplicate` will not be aggregated.
            remainder_into (K | None): The target column to move all remaining values into
                that are not covered by `deduplicate` or `aggregate`.
                If None, these values will be discarded.
            in_place (bool): Apply the operation **in-place**, modifying the original dataset.
                Default: `True`.

        Returns:
            Dataset: The grouped dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"foo": 1, "bar": "baz", "values": [1,2,3]},
            ...     {"foo": 1, "bar": "baz", "values": [4,5,6]},
            ...     {"foo": 2, "bar": "qux", "values": [7,8,9]},
            ... ])
            >>> dataset.group_by_column("foo", ("foo", "bar",), ("values",)).data
            [{'foo': 1, 'bar': 'baz', 'values': [[1, 2, 3], [4, 5, 6]]}, {'foo': 2, 'bar': 'qux', 'values': [[7, 8, 9]]}]
        """
        grouped = dict()
        remove_by_column = by not in (deduplicate + aggregate)
        for source in self.data:
            key = source.pop(by) if remove_by_column else source[by]
            target = grouped.setdefault(key, dict())

            if deduplicate:
                for k in deduplicate:
                    if k in source:
                        target[k] = source.pop(k)

            if aggregate:
                for k in aggregate:
                    if k in source:
                        target.setdefault(k, []).append(source.pop(k))

            if remainder_into is not None and source:
                target.setdefault(remainder_into, []).append(source)
        if in_place:
            self.data = list(grouped.values())
            return self
        else:
            return Dataset(grouped.values())

    def sort_by_column(
        self,
        by: K,
        *to_sort: K,
        reverse: bool = False,
        **kwargs,
    ) -> Self:
        """
        Sort the columns `to_sort` in this dataset by the values provided in the given
        `by` column. The `by` column will be sorted using `numpy.argsort`.

        Note:
            Documents in this dataset are modified **in-place**.

        Args:
            by (K): The column to sort by containing lists of sortable values.
            *to_sort (K): The fields to sort. Within each document, all fields
                must be lists of the same length as the `by` column.
            reverse (bool): If true, will sort the fields in descending order.
            **kwargs: Additional keyword arguments to pass to `numpy.argsort`.

        Raises:
            ValueError: If the length of any field does not match the length of the
                `by` column for any given document.

        Returns:
            Dataset: The sorted dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"value": [3,1,2], "other": ["c", "a", "b"]},
            ...     {"value": ['y','z','x'], "other": ["b", "c", "a"]},
            ... ])
            >>> dataset.copy(deep=True).sort_by_column("value", "other", reverse=True).data
            [{'value': [3, 2, 1], 'other': ['c', 'b', 'a']}, {'value': ['z', 'y', 'x'], 'other': ['c', 'b', 'a']}]
            >>> dataset.sort_by_column("value", "other", reverse=True, include_by_column=False).data
            [{'value': [3, 1, 2], 'other': ['c', 'b', 'a']}, {'value': ['y', 'z', 'x'], 'other': ['c', 'b', 'a']}]
        """
        # lists & dicts are mutable, so we can just manipulate the values in-place
        for idx, document in enumerate(self.data):
            len_row = len(document[by])

            order = np.argsort(document[by], **kwargs)
            if reverse:
                order = order[::-1]

            for column in to_sort:
                if len(document[column]) != len_row:
                    raise ValueError(
                        f"Column '{column}' in document {idx} has a different length than the column '{by}': {len(document[column])} != {len(document[by])}"
                    )
                document[column] = [document[column][i] for i in order]
        return self

    def sort_by(
        self,
        by: K | Callable,
        reverse: bool = False,
        in_place: bool = True,
        remove_by_column: bool = False,
    ) -> Self:
        """
        Sort the dataset by the values in the given column.

        Args:
            by (K | Callable): The column to sort by or a callable to determine the sorting key.
            reverse (bool): If true, will sort the dataset in descending order.
            in_place (bool): Apply the operation **in-place**, modifying the original dataset.
                Default: `True`.
            remove_by_column (bool): If true, will remove the column used for sorting from the dataset.
                Does not apply if `by` is a callable.

        Returns:
            Dataset: The sorted dataset.

        Examples:
            >>> dataset = Dataset([
            ...     {"value": 3, "foo": "qux"},
            ...     {"value": 1, "foo": "bar"},
            ...     {"value": 2, "foo": "baz"},
            ... ])
            >>> dataset.sort_by("value").data
            [{'value': 1, 'foo': 'bar'}, {'value': 2, 'foo': 'baz'}, {'value': 3, 'foo': 'qux'}]
            >>> dataset.sort_by("value", remove_by_column=True).data
            [{'foo': 'bar'}, {'foo': 'baz'}, {'foo': 'qux'}]
        """
        if callable(by):
            data = list(sorted(self.data, key=by, reverse=reverse))
        else:
            if remove_by_column:

                def _get_or_pop(document: dict[K, V]) -> V:
                    return document.pop(by)

            else:

                def _get_or_pop(document: dict[K, V]) -> V:
                    return document[by]

            data = list(sorted(self.data, key=_get_or_pop, reverse=reverse))

        if in_place:
            self.data = data
            return self

        return Dataset(data)

    def copy(self, deep=False) -> Self:
        """
        Return a copy of the dataset.

        Args:
            deep (bool): If True, will return a `deepcopy` of the dataset.
                Default: `False`.
        """
        return Dataset(self.data.copy()) if not deep else Dataset(deepcopy(self.data))
