import enum
import gc
import os
from hashlib import sha256
from typing import Any, Final, Generator, Iterable

import torch
from transformers import AutoConfig, BatchEncoding


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

    def items(self):
        yield from ((key, self[key]) for key in self.keys())


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


def normalize_text(document):
    return " ".join(document["text"].strip().split())


def _explode_encodings(document: dict[str, Any], encoding: BatchEncoding):
    yield from (
        document | transposed
        for transposed in transpose_dict_of_lists(encoding, iter=True)
    )


def _truncate_transition_scores(document: dict[str, dict[str, list]]) -> None:
    document["transition_scores"] = {
        key: value[document["start_token_idx"] :]
        for key, value in document.pop("transition_scores").items()
    }


def add_text_sha256(document: dict[str, str]) -> None:
    document["text_sha256"] = sha256(document["text"].encode()).hexdigest()


def _pop_or_calc_length(document: dict[str, int | list]) -> int:
    """Pop or calculate the length of the document.
    If the `length` field is present, it will be popped and returned.
    Otherwise, the length of the `input_ids` field will be returned.

    Args:
        document (dict[str, int | list]): The document to calculate the length of.

    Returns:
        int: The length of the document.
    """
    length: int | None = document.pop("length", None)
    return length if length is not None else len(document["input_ids"])
