import enum
from abc import ABC, abstractmethod
from typing import NamedTuple, Self

import numpy as np
import torch

from transition_scores.data import TransitionScores


class FeaturizedTransitionScores(NamedTuple):
    target_ids: torch.Tensor
    target_probs: torch.Tensor
    target_ranks: torch.Tensor
    top_k_indices: torch.Tensor
    top_k_probs: torch.Tensor
    intermediate_logits: torch.Tensor

    def __len__(self):
        return len(self.target_probs)


class OneDimFeatures(NamedTuple):
    size: int


class TwoDimFeatures(NamedTuple):
    height: int
    width: int


class ThreeDimFeatures(NamedTuple):
    height: int
    width: int
    depth: int


type AnyDimFeatures = OneDimFeatures | TwoDimFeatures | ThreeDimFeatures


class Slicer(ABC):
    def __init__(self, size: int, offset: int = 0):
        """
        Creates a single slice of sequences.

        Args:
            size (int): The size of the slice.
            offset (int, optional): Offset from the start of the sequence. Defaults to 0.
        """
        self.size = size
        self.offset = offset

    @abstractmethod
    def slice(self, *args, **kwargs) -> tuple[slice]: ...

    class Type(enum.Enum):
        First = "first"
        Random = "random"
        RandomMultiple = "random_multiple"
        RandomStrided = "random_strided"

        def into(self, feature_dim: AnyDimFeatures) -> Self:
            match self:
                case self.First:
                    return SliceFirst(feature_dim[0])
                case self.Random:
                    return SliceRandom(feature_dim[0])
                case self.RandomMultiple:
                    return SliceRandomMultiple(feature_dim[0], feature_dim[1])
                case self.RandomStrided:
                    return SliceRandomStrided(feature_dim[0], feature_dim[1])
                case _:
                    raise RuntimeError(f"Invalid Slicer type: {self}")


class MutliSlicer(Slicer):
    def __init__(self, size: int, multiple: int, offset: int = 0, sort: bool = False):
        """
        Creates multiple slices of sequences.

        Args:
            size (int): The size of the slices.
            multiple (int): The number of slices to create.
            offset (int, optional): Offset from the start of the sequence. Defaults to 0.
            sort (bool, optional): If true, sort the slices in ascending order.
                Otherwise, the slices are in random order. Defaults to False.
        """
        super().__init__(size, offset)
        self.multiple = multiple
        self.sort = sort

    @abstractmethod
    def slice(self, *args, **kwargs) -> tuple[slice, ...]: ...


class SliceFirst(Slicer):
    def slice(self, *_args, **_kwargs) -> tuple[slice]:
        """
        Slice the first part of the sequence starting from the offset.

        Returns:
            tuple[slice]: A tuple with a single slice.
        """
        return (slice(self.offset, self.offset + self.size),)


class SliceRandom(Slicer):
    def slice(self, length: int) -> tuple[slice]:
        """
        Randomly slice a part of the sequence starting from the offset.

        Args:
            length (int): The length of the sequence.

        Returns:
            tuple[slice]: A tuple with a single slice.
        """
        upper = length - self.offset - self.size
        if upper <= 0:
            return slice(0, self.size)
        i = np.random.randint(self.offset, upper)
        return (slice(i, i + self.size),)


class SliceRandomMultiple(MutliSlicer):
    def slice(self, length: int, *_args, **_kwargs) -> tuple[slice, ...]:
        """
        Create random slices starting from the offset.

        Args:
            length (int): The length of the sequence.

        Returns:
            tuple[slice, ...]: A tuple with the random slices.
        """
        upper = length - self.offset - self.size
        if upper <= 0 or upper <= self.offset:
            return [slice(0, self.size)]

        slices = np.arange(self.offset, upper)
        slices = np.random.choice(
            slices,
            min(upper - self.offset, self.multiple),
            replace=False,
        )

        if self.sort:
            slices = sorted(slices)

        return tuple(slice(i, i + self.size) for i in slices)


class SliceRandomStrided(MutliSlicer):
    def __init__(
        self,
        size: int,
        multiple: int,
        stride: int | None = None,
        offset=0,
        sort=False,
    ):
        """
        Randomly slice a part of the sequence starting from the offset with a given stride.

        Args:
            size (int): _description_
            multiple (int): _description_
            stride (int | None, optional): Stride of the slices. Defaults to to `size`.
            offset (int, optional): Offset from the start of the sequence. Defaults to 0.
            sort (bool, optional): If true, sort the slices in ascending order.
                Otherwise, the slices are in random order. Defaults to False.
        """
        super().__init__(size, multiple, offset, sort)
        self.stride = stride or size

    def slice(self, length: int, *_args, **_kwargs) -> tuple[slice, ...]:
        """
        Create random slices with a given stride starting from the offset.

        Args:
            length (int): The length of the sequence.

        Returns:
            tuple[slice, ...]: A tuple with the random slices.
        """
        upper = length - self.offset - self.size
        if upper <= 0 or upper <= self.offset:
            return [slice(0, self.size)]

        slices = np.arange(self.offset, upper, self.stride)
        slices = np.random.choice(
            slices,
            min(upper - self.offset, self.multiple),
            replace=False,
        )

        if self.sort:
            slices = sorted(slices)

        return tuple(slice(i, i + self.size) for i in slices)


class FeatureExtractor(ABC):
    @abstractmethod
    def __call__(self, *features: torch.Tensor, **kwargs) -> torch.Tensor: ...

    @abstractmethod
    def featurize(
        self, transition_scores: TransitionScores, slices: tuple[slice, ...]
    ) -> list[torch.Tensor]: ...

    class Type(enum.Enum):
        Likelihood = "likelihood"
        LogLikelihoodLogRankRatio = "log_likelihood_log_rank_ratio"
        LikelihoodTopkLikelihoodRatio = "likelihood_topk_likelihood_ratio"
        TopkLikelihoodLikelihoodRatio = "topk_likelihood_likelihood_ratio"
        IntermediateLogits = "intermediate_logits"

        def into(self, *args, **kwargs) -> Self:
            match self:
                case self.Likelihood:
                    return Likelihood(*args, **kwargs)
                case self.LogLikelihoodLogRankRatio:
                    return LogLikelihoodLogRankRatio(*args, **kwargs)
                case self.LikelihoodTopkLikelihoodRatio:
                    return LikelihoodTopkLikelihoodRatio(*args, **kwargs)
                case self.TopkLikelihoodLikelihoodRatio:
                    return TopkLikelihoodLikelihoodRatio(*args, **kwargs)
                case self.IntermediateLogits:
                    return IntermediateLogits(*args, **kwargs)
                case _:
                    raise RuntimeError(f"Invalid FeatureExtractor type: {self}")


class Likelihood(FeatureExtractor):
    def __call__(
        self,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        return target_probs.float()

    def featurize(self, transition_scores: TransitionScores, slices: tuple[slice, ...]):
        target_probs = torch.tensor(transition_scores.target_probs)
        return [self(target_probs[s]) for s in slices]


class LogLikelihoodLogRankRatio(FeatureExtractor):
    def __call__(
        self,
        target_probs: torch.Tensor,
        target_ranks: torch.Tensor,
        epsilon: float = 1e-8,
    ) -> torch.Tensor:
        """Compute the log-likelihood log-rank ratio.

        Args:
            target_probs (torch.Tensor): Target token probabilities.
            target_ranks (torch.Tensor): Zero-indexed target token ranks.

        Returns:
            torch.Tensor: Tensor of the same shape as target_probs and target_ranks.
        """
        return (
            torch.div(
                # probs are generally small but may approach zero due to rounding
                # in our pipeline, so we add epsilon to avoid log(0)
                torch.log(target_probs + epsilon),
                # ranks, however, are 0-indexed, so we use log1p to avoid log(0)
                # and add epsilon to avoid division by zero
                torch.log1p(target_ranks) + epsilon,
            )
            .exp()  # FIXME
            .float()
        )

    def featurize(self, transition_scores: TransitionScores, slices: tuple[slice, ...]):
        target_probs, target_ranks = (
            torch.tensor(transition_scores.target_probs),
            torch.tensor(transition_scores.target_ranks),
        )
        return [self(target_probs[s], target_ranks[s]) for s in slices]


class LikelihoodTopkLikelihoodRatio(FeatureExtractor):
    def __init__(self, k: int = 8):
        super().__init__()
        self.k = k

    def __call__(
        self,
        target_probs: torch.Tensor,
        top_k_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Target likelihood divided by top-k likelihood.

        Args:
            target_probs (torch.Tensor): Target token probabilities.
            top_k_probs (torch.Tensor): Top-k token probabilities.

        Returns:
            torch.Tensor: Tensor of the same shape as target_probs and top_k_probs.
        """
        return torch.div(target_probs, top_k_probs)

    def featurize(
        self, transition_scores: TransitionScores, slices: tuple[slice, ...]
    ) -> list[torch.Tensor]:
        target_probs, top_k_probs = (
            torch.tensor(transition_scores.target_probs),
            torch.tensor(transition_scores.top_k_probs),
        )
        return [self(target_probs[s, self.k], top_k_probs[s, self.k]) for s in slices]


class TopkLikelihoodLikelihoodRatio(LikelihoodTopkLikelihoodRatio):
    def __call__(
        self,
        target_probs: torch.Tensor,
        top_k_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Target likelihood divided by top-k likelihood.

        Args:
            target_probs (torch.Tensor): Target token probabilities.
            top_k_probs (torch.Tensor): Top-k token probabilities.

        Returns:
            torch.Tensor: Tensor of the same shape as top_k_probs.
        """
        return torch.div(top_k_probs, target_probs)


def log_likelihood_top_k_likelihood_ratio(
    target_probs: torch.Tensor,
    top_k_probs: torch.Tensor,
) -> torch.Tensor:
    """Log of the target likelihood divided by top-k likelihood.

    Args:
        target_probs (torch.Tensor): Target token probabilities.
        top_k_probs (torch.Tensor): Top-k token probabilities.

    Returns:
        torch.Tensor: Tensor of the same shape as target_probs and top_k_probs.
    """
    return torch.div(target_probs, top_k_probs).log()


def log_likelihood_log_top_k_likelihood_ratio(
    target_probs: torch.Tensor,
    top_k_probs: torch.Tensor,
) -> torch.Tensor:
    """Target log-likelihood divided by top-k log-likelihood.

    Args:
        target_probs (torch.Tensor): Target token probabilities.
        top_k_probs (torch.Tensor): Top-k token probabilities.

    Returns:
        torch.Tensor: Tensor of the same shape as target_probs and top_k_probs.
    """
    return torch.div(target_probs.log(), top_k_probs.log())


class IntermediateLogits(Likelihood):
    def __init__(self, layers: slice | None):
        super().__init__()
        self.layers = layers or slice(0, None)

    def featurize(
        self, transition_scores: TransitionScores, slices: tuple[slice, ...]
    ) -> list[torch.Tensor]:
        intermediate_logits = torch.tensor(transition_scores.intermediate_logits)
        return [self(intermediate_logits[s, self.layers]) for s in slices]
