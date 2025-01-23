from transition_scores.utils import (
    transpose_dict_of_lists,
)


class TransitionScores(dict):
    def __init__(
        self,
        target_id: int,
        target_prob: float,
        top_k_ids: list[int],
        top_k_scores: list[float],
    ):
        super().__init__(
            {
                "target_id": target_id,
                "target_prob": target_prob,
                "top_k_ids": top_k_ids,
                "top_k_scores": top_k_scores,
            }
        )

    @classmethod
    def new(cls, tuple_or_target_id: tuple | int, *args):
        if isinstance(tuple_or_target_id, tuple):
            return cls(*tuple_or_target_id)
        return cls(tuple_or_target_id, *args)


def flatten_batch_encoding_of_one(row: dict[str, list]) -> dict[str, list]:
    encoding = row.pop("encoding")[0]
    flattend = {key: [] for key in row.keys() | encoding.keys()}
    for transposed in transpose_dict_of_lists(encoding, iter=True):
        for key, [value] in row.items():
            flattend[key].append(value)
        for key, value in transposed.items():
            flattend[key].append(value)
    return flattend
