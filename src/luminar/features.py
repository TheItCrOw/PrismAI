from abc import ABC, abstractmethod
from itertools import batched
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray

from prismai_features.data import FeatureValues


class OneDimFeatures(NamedTuple):
    width: int


class TwoDimFeatures(NamedTuple):
    width: int
    height: int


class ThreeDimFeatures(NamedTuple):
    width: int
    height: int
    depth: int


type AnyDimFeatures = OneDimFeatures | TwoDimFeatures | ThreeDimFeatures


class Slicer(ABC):
    def __init__(self, size: int, sort: bool = False):
        """
        Creates a single slice of sequences.

        Args:
            size (int): The size of the slice.
        """
        self.size = size
        self.sort = sort

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size}, sort={self.sort})"

    @abstractmethod
    def slice(self, length: int, *args, **kwargs) -> NDArray: ...

    @abstractmethod
    def sample(
        self, length: int, num_samples: int, *args, **kwargs
    ) -> list[NDArray]: ...

    @classmethod
    def First(cls, size: int) -> "SliceFirst":
        return SliceFirst(size)

    @classmethod
    def Random(cls, size: int, stride: int = 1) -> "SliceRandom":
        return SliceRandom(size, stride=stride)

    @classmethod
    def RandomMultiple(
        cls,
        size: int,
        multiple: int,
        stride: int = 1,
        sort: bool = False,
        infer_slice_size: bool = False,
    ) -> "SliceRandomMultiple":
        return SliceRandomMultiple(
            size,
            multiple,
            stride=stride,
            sort=sort,
            infer_slice_size=infer_slice_size,
        )


class SliceFirst(Slicer):
    def slice(self, length: int) -> NDArray:
        """
        Slice the first part of the sequence starting from the offset.

        Returns:
            NDArray: A single slice.
        """
        return np.arange(0, min(self.size, length))

    def sample(self, length: int, _: int) -> list[NDArray]:
        return [self.slice(length)]


class SliceRandom(Slicer):
    def __init__(
        self,
        size: int,
        stride: int = 1,
        sort: bool = False,
    ):
        """
        Randomly slice a part of the sequence starting from the offset with a given stride.

        Args:
            size (int): The size of the slices.
            stride (int, optional): Stride of the slices (for sampling). Defaults to to 1.
            sort (bool, optional): If true, sort the slices in ascending order.
                Otherwise, the slices are in random order. Defaults to False.
        """
        super().__init__(size, sort)
        self.stride = stride

    def __repr__(self):
        return (
            f"{type(self).__name__}(size={self.size}, sort={self.sort})"
            if self.stride == 1
            else f"{type(self).__name__}(size={self.size}, stride={self.stride}, sort={self.sort})"
        )

    def slice(self, length: int) -> NDArray:
        """
        Randomly slice a part of the sequence starting from the offset.

        Args:
            length (int): The length of the sequence.

        Returns:
            NDArray: A single slice.
        """
        upper = length - self.size
        if upper <= 0:
            return np.arange(0, min(self.size, length))
        i = np.random.randint(0, upper)
        return np.arange(i, i + self.size)

    def sample(self, length: int, num_samples: int) -> list[NDArray]:
        """
        Create random slices starting from the offset.

        Args:
            length (int): The length of the sequence.
            num_samples (int): The number of samples to create.

        Returns:
            list[NDArray]: A list with the random slices.
        """
        upper = int(length - self.size)
        if upper <= 0:
            return [np.arange(0, min(self.size, length))]

        slices = np.arange(0, upper, self.stride)
        slices = np.random.choice(
            slices,
            min(len(slices), num_samples),
            replace=False,
        )

        if self.sort:
            slices = sorted(slices)

        return [np.arange(i, i + self.size) for i in slices]


class SliceRandomMultiple(SliceRandom):
    def __init__(
        self,
        size: int,
        multiple: int,
        stride: int = 1,
        sort: bool = False,
        infer_slice_size: bool = False,
    ):
        if infer_slice_size:
            if size % multiple != 0:
                raise ValueError(
                    f"Size {size} must be a divisible by multiple {multiple}"
                )
            size //= multiple

        super().__init__(size, stride=stride, sort=sort)
        self.multiple = multiple

    def __repr__(self):
        return (
            f"{type(self).__name__}(size={self.size}, multiple={self.multiple}, sort={self.sort})"
            if self.stride == 1
            else f"{type(self).__name__}(size={self.size}, multiple={self.multiple}, stride={self.stride}, sort={self.sort})"
        )

    def slice(self, length: int) -> NDArray:
        return np.array(super().sample(length, self.multiple))

    def sample(self, length: int, num_samples: int) -> list[NDArray]:
        if num_samples == 1 or (length - self.size) <= 0:
            return [self.slice(length)]

        # non_overlapping_slices = (length / self.stride) // (self.size * self.multiple)
        # if non_overlapping_slices < num_samples:
        #     warnings.warn(
        #         f"Sequence with length {length} can only contain {non_overlapping_slices} non-overlapping slices, but you set num_samples={num_samples}."
        #         + (
        #             " Sorting is disabled, so we may not get samples with overlapping slices."
        #             if not self.sort
        #             else " Sorting is ENABLED: we WILL get samples with overlapping slices! "
        #         )
        #         + " Reduce num_samples or slice size to avoid this warning."
        #     )
        samples = super().sample(length, self.multiple * num_samples)
        samples = [np.array(batch) for batch in batched(samples, self.multiple)]
        return samples[:num_samples]


class FeatureExtractor(ABC):
    @abstractmethod
    def __call__(self, *args: torch.Tensor, **kwargs) -> torch.Tensor: ...

    def __repr__(self):
        return f"{type(self).__name__}()"

    @abstractmethod
    def featurize(
        self, features: FeatureValues, slices: tuple[slice, ...] | NDArray
    ) -> torch.Tensor: ...

    @classmethod
    def Likelihood(cls) -> "Likelihood":
        return Likelihood()

    @classmethod
    def LogLikelihoodLogRankRatio(cls) -> "LogLikelihoodLogRankRatio":
        return LogLikelihoodLogRankRatio()

    @classmethod
    def LikelihoodTopkLikelihoodRatio(
        cls, top_k: int
    ) -> "LikelihoodTopkLikelihoodRatio":
        return LikelihoodTopkLikelihoodRatio(top_k)

    @classmethod
    def TopkLikelihoodLikelihoodRatio(
        cls, top_k: int
    ) -> "TopkLikelihoodLikelihoodRatio":
        return TopkLikelihoodLikelihoodRatio(top_k)

    @classmethod
    def IntermediateLikelihood(
        cls, last_n: int | None = None
    ) -> "IntermediateLikelihood":
        return IntermediateLikelihood(last_n)


class Likelihood(FeatureExtractor):
    def __call__(
        self,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        return target_probs.float()  # .mul(2).sub(1.0)

    def featurize(self, ts: FeatureValues, slices: slice | list[slice]) -> torch.Tensor:
        target_probs = torch.tensor(ts.target_probs)[slices].flatten()

        return self(target_probs)


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
            .neg()
            .float()
        )

    def featurize(self, ts: FeatureValues, slices: slice | list[slice]) -> torch.Tensor:
        target_probs = torch.tensor(ts.target_probs)[slices].flatten()
        target_ranks = torch.tensor(ts.target_ranks)[slices].flatten()

        return self(target_probs, target_ranks)


class LikelihoodTopkLikelihoodRatio(FeatureExtractor):
    def __init__(self, top_k: int = 8, normalize: bool = True):
        super().__init__()
        self.top_k = top_k
        self.normalize = normalize

    def __repr__(self):
        return f"{type(self).__name__}(top_k={self.top_k})"

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
        return (
            torch.div(
                target_probs, target_probs + top_k_probs + 1e-8
            )  # .mul(2).sub(1.0)
        )

    def featurize(self, ts: FeatureValues, slices: slice | list[slice]) -> torch.Tensor:
        target_probs = torch.tensor(ts.target_probs)[slices].view(-1, 1)
        top_k_probs = torch.tensor(ts.top_k_probs)[slices]
        top_k_probs = top_k_probs[..., : self.top_k].view(-1, self.top_k)

        return self(target_probs, top_k_probs)


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
        return torch.div(
            top_k_probs, target_probs + top_k_probs + 1e-8
        )  # .mul(2).sub(1.0)


class IntermediateLikelihood(FeatureExtractor):
    def __init__(self, last_n: int | None = None):
        super().__init__()
        self.last_n = last_n

    def __repr__(self):
        return f"{type(self).__name__}(last_n={self.last_n})"

    def __call__(
        self,
        intermediate_probs: torch.Tensor,
    ) -> torch.Tensor:
        return intermediate_probs.float()  # .mul(2).sub(1.0)

    def featurize(self, ts: FeatureValues, slices: slice | list[slice]) -> torch.Tensor:
        logits = torch.tensor(ts.intermediate_likelihoods)[slices]
        last_n = self.last_n or logits.size(-1)
        logits = logits[..., -last_n:].view(-1, last_n)

        return self(logits)


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
