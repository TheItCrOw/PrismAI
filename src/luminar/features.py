import enum
from abc import ABC, abstractmethod
from typing import NamedTuple, Self

import numpy as np
import torch

from transition_scores.data import TransitionScores


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


class FeatureSelection(ABC):
    def __init__(
        self,
        size: int | tuple[int, ...],
        offset: int = 1,
    ):
        match size:
            case (number,) | number if isinstance(number, int):
                self.size = OneDimFeatures(number)
            case (h, w):
                self.size = TwoDimFeatures(h, w)
            case (h, w, d):
                self.size = ThreeDimFeatures(h, w, d)
            case _:
                raise ValueError(f"Invalid feature size {size}")

        self.offset = offset

    @abstractmethod
    def __call__(
        self,
        *features: np.ndarray,
        **kwargs,
    ) -> list[np.ndarray]:
        pass

    @classmethod
    def first(cls, size: OneDimFeatures | TwoDimFeatures | ThreeDimFeatures) -> Self:
        return FeatureSelectionFirst(size)

    @classmethod
    def random(cls, size: OneDimFeatures | TwoDimFeatures | ThreeDimFeatures) -> Self:
        return FeatureSelectionRandom(size)

    @classmethod
    def random_no_overlap(cls, size: TwoDimFeatures | ThreeDimFeatures) -> Self:
        return FeatureSelectionRandomNoOverlap(size)

    def effective_size(self) -> int:
        match self.size:
            case (size,):
                return size
            case (a, b) | (a, b, _):
                return a * b

    def required_size(self) -> int:
        return self.effective_size() + self.offset

    def effective_shape(self) -> OneDimFeatures | TwoDimFeatures | ThreeDimFeatures:
        return self.size

    class Type(enum.StrEnum):
        First = "First"
        Random = "Random"
        RandomNoOverlap = "RandomNoOverlap"

        def into(
            self, size: OneDimFeatures | TwoDimFeatures | ThreeDimFeatures
        ) -> "FeatureSelection":
            match self:
                case FeatureSelection.Type.First:
                    return FeatureSelectionFirst(size)
                case FeatureSelection.Type.Random:
                    return FeatureSelectionRandom(size)
                case FeatureSelection.Type.RandomNoOverlap:
                    return FeatureSelectionRandomNoOverlap(size)
                case _:
                    raise RuntimeError


class FeatureSelectionFirst(FeatureSelection):
    def __call__(
        self,
        *features: np.ndarray,
    ) -> list[np.ndarray]:
        match self.size:
            case (size,):
                return [
                    torch.tensor(feature[self.offset : self.offset + size])
                    for feature in features
                ]
            case (h, w) | (h, w, _):
                slices = np.arange(self.offset, len(features[0]) - w, w)[:h]
                slices = slices.reshape(-1, 1).repeat(w, 1)
                slices = slices + np.arange(w).reshape(1, -1)
                return [torch.tensor(feature[slices]) for feature in features]
            case _:
                raise RuntimeError(f"Invalid slice size: {size}")


class FeatureSelectionRandom:
    def __init__(
        self,
        size: OneDimFeatures | TwoDimFeatures | ThreeDimFeatures,
        offset: int = 1,
        sort: bool = True,
        flatten: bool = False,
    ):
        super().__init__(size, offset)
        self.sort = sort
        self.flatten = flatten

    def __call__(
        self,
        *features: np.ndarray,
    ) -> list[np.ndarray]:
        length = len(features[0])
        match self.size:
            case (size,):
                offset = np.random.randint(self.offset, length - self.offset - size)
                return [
                    torch.tensor(feature[offset : offset + size])
                    for feature in features
                ]
            case (h, w) | (h, w, _):
                slices = np.arange(self.offset, length - self.offset - w)
                slices = np.random.choice(slices, h, replace=False)

                if self.sort:
                    slices = np.sort(slices)

                slices = slices.reshape(-1, 1).repeat(w, 1)
                slices = slices + np.arange(w).reshape(1, -1)

                features = [torch.tensor(feature[slices]) for feature in features]
                if self.flatten:
                    features = [
                        feature.flatten()
                        if feature.ndim == 2
                        else feature.reshape(-1, feature.shape[-1])
                        for feature in features
                    ]
                return features
            case _:
                raise RuntimeError(f"Invalid slice size: {size}")

    def effective_shape(self) -> OneDimFeatures | TwoDimFeatures | ThreeDimFeatures:
        match self.flatten, self.size:
            case True, (w, h) | (w, h, _):
                return TwoDimFeatures(w * h, *self.size[2:])
            case _, shape:
                return shape


class FeatureSelectionRandomNoOverlap(FeatureSelectionRandom):
    def __init__(
        self,
        size: TwoDimFeatures | ThreeDimFeatures,
        offset: int = 1,
        sort: bool = True,
        flatten: bool = False,
    ):
        super().__init__(size, offset=offset, sort=sort, flatten=flatten)
        match self.size:
            case (_,):
                raise ValueError(f"{type(self).__name__} does not support 1D features")
            case (w, h) | (w, h, _):
                self.w = w
                self.h = h

    def __call__(
        self,
        *features: np.ndarray,
    ) -> list[torch.Tensor]:
        length = len(features[0])
        slices = np.arange(self.offset, length - self.offset - self.w, self.w)
        slices = np.random.choice(slices, self.h, replace=False)

        if self.sort:
            slices = np.sort(slices)

        slices = slices.reshape(-1, 1).repeat(self.w, 1)
        slices = slices + np.arange(self.w).reshape(1, -1)

        features = [torch.tensor(feature[slices]) for feature in features]
        if self.flatten:
            features = [
                feature.flatten()
                if feature.ndim == 2
                else feature.view(-1, feature.shape[-1])
                for feature in features
            ]
        return features


class FeatureAlgorithm(ABC):
    def __init__(
        self,
        feature_selection: FeatureSelection,
    ):
        self.feature_selection = feature_selection

    @abstractmethod
    def __call__(self, *features: np.ndarray, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def featurize(self, transition_scores: TransitionScores) -> torch.Tensor:
        pass

    class Type(enum.StrEnum):
        Likelihood = Default = "Likelihood"
        LLR = LogLikelihoodLogRankRatio = "Log-Likelihood Log-Rank Ratio"
        LTR = LikelihoodTopkLikelihoodRatio = "Likelihood Top-k Likelihood Ratio"
        TLR = TopkLikelihoodLikelihoodRatio = "Top-k Likelihood Likelihood Ratio"
        LTS = LikelihoodTopkStack = "Likelihood Top-k Stack"

        def into(self, *args, **kwargs) -> "FeatureAlgorithm":
            match self:
                case self.Likelihood:
                    return Likelihood(*args, **kwargs)
                case self.LLR:
                    return LogLikelihoodLogRankRatio(*args, **kwargs)
                case self.LTR:
                    return LikelihoodTopkLikelihoodRatio(*args, **kwargs)
                case self.TLR:
                    return TopkLikelihoodLikelihoodRatio(*args, **kwargs)
                case self.LTS:
                    raise NotImplementedError(
                        f"{FeatureAlgorithm.__name__}({self.value}) is not implemented yet"
                    )
                    # return LikelihoodTopkStack(*args, **kwargs)
                case _:
                    raise RuntimeError


class Likelihood(FeatureAlgorithm):
    def __call__(
        self,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the likelihood.

        Args:
            target_probs (torch.Tensor): Target token probabilities.

        Returns:
            torch.Tensor: Tensor of the same shape as target_probs.
        """
        return target_probs.float()

    def featurize(self, transition_scores: TransitionScores):
        (target_probs,) = self.feature_selection(transition_scores.target_probs)
        return self(target_probs)


class LogLikelihoodLogRankRatio(FeatureAlgorithm):
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
                # probs are generally small but may approach zero due to rounding,
                # so we add epsilon to avoid log(0)
                torch.log(target_probs + epsilon),
                # ranks, however, are 0-indexed, so we use log1p to avoid log(0)
                # and add epsilon to avoid division by zero
                torch.log1p(target_ranks) + epsilon,
            )
            .exp()
            .float()
        )

    def featurize(self, transition_scores: TransitionScores):
        target_probs, target_ranks = self.feature_selection(
            transition_scores.target_probs,
            transition_scores.target_ranks,
        )
        return self(target_probs, target_ranks)


class LikelihoodTopkLikelihoodRatio(FeatureAlgorithm):
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

    def featurize(self, transition_scores: TransitionScores):
        target_probs, top_k_probs = self.feature_selection(
            transition_scores.target_probs,
            transition_scores.top_k_probs,
        )
        return self(target_probs, top_k_probs)


class TopkLikelihoodLikelihoodRatio(FeatureAlgorithm):
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

    def featurize(self, transition_scores: TransitionScores):
        target_probs, top_k_probs = self.feature_selection(
            transition_scores.target_probs,
            transition_scores.top_k_probs,
        )
        return self(target_probs, top_k_probs)


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
