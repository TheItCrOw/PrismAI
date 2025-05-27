import argparse
import dataclasses
import json
from hashlib import sha256
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Generator,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    TextIO,
)

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from attr import dataclass
from datasets import Dataset, DatasetDict
from matplotlib.image import AxesImage
from numpy.typing import NDArray
from transformers import Trainer  # type: ignore

acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")
roc_auc = evaluate.load("roc_auc")


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
    dataset_unmatched = dataset.filter(
        lambda _id: _id not in ids_matched, input_columns=["id_source"]
    )

    return dataset_matched, dataset_unmatched


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


class ConvolutionalLayerSpec(NamedTuple):
    channels: int
    kernel_size: int | tuple[int, int]
    stride: int = 1

    @property
    def kernel_size_1d(self):
        if isinstance(self.kernel_size, int):
            return self.kernel_size
        return self.kernel_size[0]

    @property
    def kernel_size_2d(self):
        if isinstance(self.kernel_size, int):
            return (self.kernel_size, self.kernel_size)
        return self.kernel_size

    @property
    def padding(self) -> int:
        return (self.kernel_size_1d - 1) // 2

    def __repr__(self):
        return repr(tuple(self))


type ProjectionDim = Optional[int | tuple[int, int] | tuple[int, int, int]]


DEFAULT_CONV_LAYER_SHAPES: Final[tuple[ConvolutionalLayerSpec, ...]] = (
    ConvolutionalLayerSpec(32, 5),
    ConvolutionalLayerSpec(64, 5),
    ConvolutionalLayerSpec(32, 3),
)


class LuminarTrainingConfig(argparse.Namespace):
    feature_len: int
    feature_dim: tuple[int, int]
    feature_type: Literal["intermediate_likelihoods"]
    feature_model: Literal["gpt2"]
    feature_selection: Literal["first"]

    agent: str
    domain: str
    other_agents: tuple[str, ...] | None = None
    datset_config_name: str
    dataset_split_name: str

    conv_layer_shapes: tuple[ConvolutionalLayerSpec, ...] = DEFAULT_CONV_LAYER_SHAPES
    projection_dim: ProjectionDim = (1024, 32)

    max_epochs: int = 25
    learning_rate: float = 5e-4
    gradient_clip_val: float = 1.0
    train_batch_size: int = 32
    eval_batch_size: int = 1024
    warmup_ratio: float = 1.0
    seed: int = 42

    def json(self, /, **kwargs) -> str:
        kwargs.setdefault("sort_keys", True)
        kwargs.setdefault("indent", 4)
        return json.dumps(vars(self), **kwargs)

    def dump(self, fp: TextIO, /, **kwargs) -> None:
        fp.write(self.json(**kwargs))

    def hash(self, trim: int | None = None) -> str:
        config_hash = sha256(self.json().encode()).hexdigest()
        return config_hash[:trim]

    def name(self) -> str:
        return "-".join(
            (
                self.domain,
                self.agent,
                self.feture_model,
                str(self.feature_len),
                self.hash(10),
            )
        )

    def asdict(self) -> dict[str, Any]:
        return vars(self)


def save_model(
    trainer: Trainer,
    config: dict | LuminarTrainingConfig,
    root: str | Path = Path.home() / "Projects/PrismAI/models/luminar_cnn",
    infix: str = "",
    suffix: str = "",
) -> Path:
    if isinstance(config, dict):
        config = LuminarTrainingConfig(**config)

    config_str, config_hash = config.str_and_hash(10)

    path = Path(root) / infix / config.name() / suffix

    trainer.save_model(str(path))

    with (path / "pytorch_model.bin").open("wb") as fp:
        torch.save(trainer.model, fp)

    with (path / "config.json").open("w") as fp:
        config.dump(fp)

    with (path / "trainer_state.json").open("w") as fp:
        json.dump(dataclasses.asdict(trainer.state), fp, indent=4)

    return path


def compute_scores(preds: NDArray, labels: NDArray, suffix=""):
    f1_score_each = f1.compute(predictions=preds, references=labels, average=None)
    f1_score_weighted = f1.compute(
        predictions=preds, references=labels, average="weighted"
    )
    acc_score = acc.compute(predictions=preds, references=labels)
    roc_auc_score = roc_auc.compute(prediction_scores=preds, references=labels)

    return {
        f"f1_human{suffix}": f1_score_each["f1"][0],  # type: ignore
        f"f1_ai{suffix}": f1_score_each["f1"][1],  # type: ignore
        f"f1_weighted{suffix}": f1_score_weighted["f1"],  # type: ignore
        f"accuracy{suffix}": acc_score["accuracy"],  # type: ignore
        f"roc_auc{suffix}": roc_auc_score["roc_auc"],  # type: ignore
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    labels = np.array(labels)
    scores = 1 / (1 + np.exp(-np.array(logits)))

    metrics = compute_scores(scores > 0.5, labels)

    gt_0 = np.sum(labels == 0)
    gt_1 = np.sum(labels == 1)

    if gt_0 == gt_1:
        # dataset is balanced, use the median of all scores as threshold
        threshold = np.median(scores)
        metrics |= compute_scores(scores > threshold, labels, "_median")
        metrics["threshold_median"] = threshold
    elif gt_0 > 0 < gt_1:
        # dataset is unbalanced, use the midpoint between the means of the two classes as threshold
        threshold = float(scores[labels == 0].mean() + scores[labels == 1].mean()) / 2
        metrics |= compute_scores(scores > threshold, labels, "_mean")
        metrics["threshold_mean"] = threshold
    else:
        # only one class is present
        # TODO?
        pass

    metrics["ground_truth_human"] = gt_0
    metrics["ground_truth_ai"] = gt_1

    return metrics
