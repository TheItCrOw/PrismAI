import argparse
import dataclasses
import json
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
from transformers import Trainer  # type: ignore


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
