import torch

from transition_scores.data import FeatureValues


def log_likelihood_log_rank_ratio(
    target_probs: torch.Tensor, target_ranks: torch.Tensor
) -> float:
    return -target_probs.log().sum().div(target_ranks.log1p().sum()).item()


def llr_from_transition_scores(transition_scores: FeatureValues) -> float:
    target_probs = torch.tensor(transition_scores.target_probs)
    target_ranks = torch.tensor(transition_scores.target_ranks)
    mask = target_probs.ne(0.0)
    return log_likelihood_log_rank_ratio(
        target_probs[mask],
        target_ranks[mask],
    )
