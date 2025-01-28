from typing import Generator, Iterable

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
