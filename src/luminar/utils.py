import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from attr import dataclass
from datasets import Dataset, DatasetDict
from matplotlib.image import AxesImage
from numpy.typing import NDArray


def get_matched_ids(dataset: Dataset, agent_set: set[str]) -> list[str]:
    ids_agent: set[str] = set(
        dataset.filter(lambda a: a in agent_set, input_columns=["agent"])["id_source"]
    )
    ids_human: set[str] = set(
        dataset.filter(lambda a: a == "human", input_columns=["agent"])["id_source"]
    )
    return list(ids_agent.intersection(ids_human))


def get_matched_datasets(
    dataset: Dataset,
    *agents: str,
    eval_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 42,
) -> DatasetDict:
    agent_set: set[str] = set(agents) if isinstance(agents[0], str) else set(agents[0])
    ids_matched = np.array(get_matched_ids(dataset, agent_set), dtype=str)
    agent_set.add("human")

    ids_matched.sort()
    np.random.seed(seed)
    np.random.shuffle(ids_matched)

    eval_offset = int(len(ids_matched) * (1 - eval_split - test_split))
    test_offset = int(len(ids_matched) * (1 - test_split))
    ids_train, ids_eval, ids_test = map(
        set, np.array_split(ids_matched, [eval_offset, test_offset])
    )

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


type DatasetUnmatched = Dataset
type DatasetDictTrainEvalTest = DatasetDict


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
    ids_matched = np.array(get_matched_ids(dataset, agent_set), dtype=str)
    agent_set.add("human")

    ids_matched.sort()
    np.random.seed(seed)
    np.random.shuffle(ids_matched)

    ids_matched_splits = np.array_split(ids_matched, num_splits)

    dataset_splits = []
    for _ in range(num_splits):
        ids_eval = set(ids_matched_splits[:eval_splits])
        ids_test = set(ids_matched_splits[eval_splits : eval_splits + test_splits])
        ids_train = set(ids_matched_splits[eval_splits + test_splits :])

        dataset_splits.append(
            DatasetDict(
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
                }
            )
        )

        # Rotate the splits for the next iteration
        ids_matched_splits.append(ids_matched_splits.pop(0))

    return dataset_splits, dataset.filter(
        lambda _id: _id not in ids_matched, input_columns=["id_source"]
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


def visualize_features(features: NDArray) -> AxesImage:
    fig = plt.imshow(
        features,
        cmap=sns.cubehelix_palette(as_cmap=True),
        vmin=min(0.0, features.min()),
        vmax=max(1.0, features.max()),
    )
    fig.axes.set_axis_off()
    plt.tight_layout()
    return fig
