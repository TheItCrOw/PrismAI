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
