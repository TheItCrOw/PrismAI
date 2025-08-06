import json
from argparse import Namespace
from dataclasses import asdict as asdict_
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
    NamedTuple,
    Optional,
    TextIO,
)

import torch
from torch.utils.data.dataset import Dataset
from transformers import Trainer, TrainingArguments  # type: ignore
from ulid import ulid

from luminar.utils.misc import docs


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
        """Calculate the padding size to maintain the input size."""
        return (self.kernel_size_1d - 1) // 2

    def __repr__(self):
        return repr(tuple(self))


DEFAULT_CONV_LAYER_SHAPES: Final[tuple[ConvolutionalLayerSpec, ...]] = (
    ConvolutionalLayerSpec(32, 5),
    ConvolutionalLayerSpec(64, 5),
    ConvolutionalLayerSpec(32, 3),
)


type ProjectionDim = Optional[int | tuple[int, int] | tuple[int, int, int]]


class BaseConfig:
    def __init__(self, *args, **kwargs):
        type_dict = type(self).__dict__
        fields: tuple[str, ...] = tuple(type_dict["__annotations__"])

        # find all arguments that are not in kwargs and set them from args (if any)
        missing = [field for field in fields if field not in kwargs]
        for key, value in zip(missing, args):
            kwargs[key] = value

        # then set all fields in order, retaining defaults
        for key in fields:
            setattr(self, key, kwargs.pop(key, type_dict.get(key, None)))

        # finally, set any additional values that were not in the fields
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.__post_init__()

    def __post_init__(self):
        pass

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in vars(self).items())})"

    def __contains__(self, key):
        return key in vars(self)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get[T](self, key: str, default: T | None = None) -> T | None:
        return getattr(self, key, default)

    def json(self, /, **kwargs) -> str:
        kwargs.setdefault("sort_keys", True)
        kwargs.setdefault("indent", 4)
        return json.dumps(vars(self), **kwargs)

    def dump(self, fp: TextIO, /, **kwargs) -> None:
        fp.write(self.json(**kwargs))

    def hash(self, trim: int | None = None) -> str:
        config_hash = sha256(self.json().encode()).hexdigest()
        return config_hash[:trim]

    def asdict(self) -> dict[str, Any]:
        return vars(self)


@dataclass
class LuminarTrainingConfig(Namespace):
    feature_len: int
    feature_dim: tuple[int, int]
    feature_type: Literal["intermediate_likelihoods"]
    feature_model: Literal["gpt2"]
    feature_selection: Literal["first"]

    agent: str
    domain: str
    datset_config_name: str
    dataset_split_name: str
    other_agents: tuple[str, ...] | None = None

    projection_dim: int | None = None
    conv_layer_shapes: tuple[ConvolutionalLayerSpec, ...] = DEFAULT_CONV_LAYER_SHAPES
    feed_forward_dim: int | tuple[int, ...] | None = (1024, 32)
    rescale_features: bool = True

    seed: int = 42

    run_ulid: str = ulid()

    @docs(TrainingArguments)
    def training_arguments(
        self,
        train_dataset_size: int,
        /,
        **kwargs,
    ) -> TrainingArguments:
        kwargs.setdefault("output_dir", f"./logs//{ulid()}")
        kwargs.setdefault("per_device_train_batch_size", 32)
        kwargs.setdefault("per_device_eval_batch_size", 1024)
        kwargs.setdefault("learning_rate", 5e-4)
        kwargs.setdefault("num_train_epochs", 10)

        train_batch_size = kwargs["per_device_train_batch_size"]
        steps_per_epoch = train_dataset_size // train_batch_size
        eval_steps = steps_per_epoch // 5

        return TrainingArguments(
            **kwargs,
            warmup_steps=steps_per_epoch,
            logging_steps=eval_steps,
            load_best_model_at_end=True,
            eval_strategy="steps",
            eval_steps=eval_steps,
            eval_delay=steps_per_epoch,
            save_strategy="steps",
            save_steps=eval_steps,
            seed=self.seed,
        )


def save_model(
    path: Path,
    trainer: Trainer,
    config: dict | LuminarTrainingConfig,
):
    if isinstance(config, dict):
        config = LuminarTrainingConfig(**config)

    trainer.save_model(str(path))

    with (path / "pytorch_model.bin").open("wb") as fp:
        torch.save(trainer.model, fp)

    with (path / "config.json").open("w") as fp:
        config.dump(fp)

    with (path / "trainer_state.json").open("w") as fp:
        json.dump(asdict_(trainer.state), fp, indent=4)


@dataclass
class LuminarSequenceTrainingConfig(Namespace):
    feature_len: int = 512
    num_intermediate_likelihoods: int = 13  # Default gpt2 with 13 hidden layers

    apply_delta_augmentation: bool = False
    apply_product_augmentation: bool = False

    conv_layer_shapes: tuple[ConvolutionalLayerSpec, ...] = DEFAULT_CONV_LAYER_SHAPES
    projection_dim: int = 32
    lstm_hidden_dim: int = 64
    lstm_layers: int = 1
    stack_spans: int = 0

    # If you want to merge two hf datasets to a larger dataset, use [DATASET_1]___[DATASET_2] etc.
    hf_dataset : str = "liberi-luminaris/Ghostbuster-encoded-gpt2"
    dataset_root_path: str = "/storage/projects/stoeckel/prismai/encoded/fulltext/"
    models_root_path: str = (
        "/storage/projects/boenisch/PrismAI/models/luminar_sequence/"
    )
    # The domain (among others) decides over the dataset - if you want to merge domains to form
    # a larger dataset, use [DOMAIN_1]___[DOMAIN_2] etc.
    domain: str = None
    agent: str = "gpt_4o_mini_gemma2_9b"
    feature_agent: str = "gpt2_512"

    max_epochs: int = 100
    batch_size: int = 128
    early_stopping_patience: int = 8
    rescale_features: bool = False
    kfold: int = 3
    learning_rate: float = 6e-4
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
                self.feature_agent,
                str(self.feature_len),
                self.hash(10),
            )
        )

    def asdict(self) -> dict[str, Any]:
        return vars(self)


class LuminarSequenceDataset(Dataset):
    def __init__(self, dataset, feature_key="features"):
        self.samples = []
        for example in dataset:
            spans = example["sentence_token_spans"]
            features = torch.tensor(example[feature_key])
            labels = example["span_labels"]
            self.samples.append(
                {"features": features, "sentence_spans": spans, "span_labels": labels}
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]