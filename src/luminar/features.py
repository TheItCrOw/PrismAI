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


class FeatureSelection(ABC):
    def __init__(self, size: OneDimFeatures | TwoDimFeatures | ThreeDimFeatures):
        self.size = size

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


class FeatureSelectionFirst(FeatureSelection):
    def __call__(
        self,
        *features: np.ndarray,
        offset: int = 1,
    ) -> list[np.ndarray]:
        match self.size:
            case (size,):
                return [feature[offset : offset + size] for feature in features]
            case (h, w) | (h, w, _):
                slices = np.arange(offset, len(features[0]) - w, w)[:h]
                slices = slices.reshape(-1, 1).repeat(w, 1)
                slices = slices + np.arange(w).reshape(1, -1)
                return [feature[slices] for feature in features]
            case _:
                raise RuntimeError(f"Invalid slice size: {size}")


class FeatureSelectionRandom:
    def __call__(
        self,
        size: OneDimFeatures | TwoDimFeatures | ThreeDimFeatures,
        *features: np.ndarray,
        offset: int = 1,
        sort: bool = True,
        flatten: bool = False,
    ) -> list[np.ndarray]:
        length = len(features[0])
        match self.size:
            case (size,):
                offset = np.random.randint(offset, length - offset - size)
                return [feature[offset : offset + size] for feature in features]
            case (h, w) | (h, w, _):
                slices = np.arange(offset, length - offset - w)
                slices = np.random.choice(slices, h, replace=False)

                if sort:
                    slices = np.sort(slices)

                slices = slices.reshape(-1, 1).repeat(w, 1)
                slices = slices + np.arange(w).reshape(1, -1)

                features = [feature[slices] for feature in features]
                if flatten:
                    features = [
                        feature.flatten()
                        if feature.ndim == 2
                        else feature.reshape(-1, feature.shape[-1])
                        for feature in features
                    ]
                return features
            case _:
                raise RuntimeError(f"Invalid slice size: {size}")


class FeatureSelectionRandomNoOverlap(FeatureSelection):
    def __init__(
        self,
        size: TwoDimFeatures | ThreeDimFeatures,
    ):
        super().__init__(size)
        match self.size:
            case (_,):
                raise ValueError(f"{type(self).__name__} does not support 1D features")
            case (w, h) | (w, h, _):
                self.w = w
                self.h = h

    def __call__(
        self,
        *features: np.ndarray,
        offset: int = 1,
        sort: bool = True,
        flatten: bool = False,
    ):
        length = len(features[0])
        slices = np.arange(offset, length - offset - self.w, self.w)
        slices = np.random.choice(slices, self.h, replace=False)

        if sort:
            slices = np.sort(slices)

        slices = slices.reshape(-1, 1).repeat(self.w, 1)
        slices = slices + np.arange(self.w).reshape(1, -1)

        features = [feature[slices] for feature in features]
        if flatten:
            features = [
                feature.flatten()
                if feature.ndim == 2
                else feature.reshape(-1, feature.shape[-1])
                for feature in features
            ]
        return features


class FeatureType(enum.StrEnum):
    LLR = LogLikelihoodLogRankRatio = "Log-Likelihood Log-Rank Ratio"
    LTR = LogLikelihoodTopkLikelihoodRatio = "Log-Likelihood Top-k Log-Likelihood Ratio"
    LTS = LikelihoodTopkStack = "Likelihood Top-k Stack"


class FeatureAlgorithm(ABC):
    @abstractmethod
    def __call__(self, *features: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def featurize(self, transition_scores: TransitionScores) -> np.ndarray:
        pass


class LogLikelihoodLogRankRatio(FeatureAlgorithm):
    def __init__(
        self,
        feature_selection: FeatureSelection,
    ):
        self.feature_selection = feature_selection

    def __call__(
        self,
        target_probs: np.ndarray,
        target_ranks: np.ndarray,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Compute the log-likelihood log-rank ratio.

        Args:
            target_probs (np.ndarray): Target token probabilities.
            target_ranks (np.ndarray): Zero-indexed target token ranks.

        Returns:
            np.ndarray: Tensor of the same shape as target_probs and target_ranks.
        """
        return -np.true_divide(
            # probs are generally small but never zero, so log(x) is safe
            np.log(target_probs),
            # ranks, however, are 0-indexed, so we use log1p to avoid log(0)
            # and add epsilon to avoid division by zero
            np.log1p(target_ranks) + epsilon,
        )

    def featurize(self, transition_scores: TransitionScores):
        target_probs, target_ranks = self.feature_selection(
            transition_scores.target_probs,
            transition_scores.target_ranks,
        )
        return self(target_probs, target_ranks)


def likelihood_top_k_likelihood_ratio(
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
