from typing import TypedDict

import evaluate
import numpy as np
from numpy.typing import NDArray
from transformers.trainer_utils import EvalPrediction
from ulid import ulid


def compute_scores(
    scores: NDArray, threshold: float, labels: NDArray, suffix=""
) -> dict[str, float]:
    # distributed evaluation requires a unique experiment ID
    experiment_id = ulid()
    evaluate_f1 = evaluate.load("f1", experiment_id=experiment_id)
    evaluate_acc = evaluate.load("accuracy", experiment_id=experiment_id)
    evaluate_roc_auc = evaluate.load("roc_auc", experiment_id=experiment_id)

    preds = (scores >= threshold).astype(int)
    f1_score_weighted = evaluate_f1.compute(
        predictions=preds, references=labels, average="weighted"
    )
    acc_score = evaluate_acc.compute(predictions=preds, references=labels)
    roc_auc_score = evaluate_roc_auc.compute(
        prediction_scores=scores, references=labels
    )
    scores = {
        f"f1_weighted{suffix}": float(f1_score_weighted["f1"]),  # type: ignore
        f"accuracy{suffix}": float(acc_score["accuracy"]),  # type: ignore
        f"roc_auc{suffix}": float(roc_auc_score["roc_auc"]),  # type: ignore
    }

    f1_score_each = evaluate_f1.compute(
        predictions=preds, references=labels, average=None
    )
    try: 
        scores |= {
            f"f1_human{suffix}": float(f1_score_each["f1"][0]),  # type: ignore
            f"f1_ai{suffix}": float(f1_score_each["f1"][1]),  # type: ignore
        }
    except TypeError:
        UserWarning(
            "evaluate(f1) with average=None returned a single value, "
            "indicating only one class is present in the dataset."
        )
        if labels[0] == 0:
            scores |= {
                f"f1_human{suffix}": float(f1_score_each["f1"]),  # type: ignore
            }
        else:
            scores |= {
                f"f1_ai{suffix}": float(f1_score_each["f1"]),  # type: ignore
            }

    return scores


class LuminarEvaluationMetrics(TypedDict):
    f1_weighted: float
    accuracy: float
    roc_auc: float
    f1_human: float
    f1_ai: float
    n_samples: int


class Balanced(LuminarEvaluationMetrics):
    f1_weighted_median: float
    accuracy_median: float
    roc_auc_median: float
    f1_human_median: float
    f1_ai_median: float
    threshold_median: float


class Unbalanced(LuminarEvaluationMetrics):
    f1_weighted_mean: float
    accuracy_mean: float
    roc_auc_mean: float
    f1_human_mean: float
    f1_ai_mean: float
    threshold_mean: float
    n_samples_human: int
    n_samples_ai: int


def compute_metrics(
    eval_pred: EvalPrediction,
) -> LuminarEvaluationMetrics | Balanced | Unbalanced:
    """Calculated weigthed and class-specific F1 scores, accuracy, ROC AUC, and class distribution for a threshold of 0.5.
    In addition, we compute best-guess thresholds based on the dataset balance:
    - If the dataset is balanced, we use the median of all scores as threshold.
    - If the dataset is unbalanced, we use the midpoint between the means of the two classes as threshold.
    - If only one class is present, no additional threshold is considered.

    Args:
        eval_pred: Tuple of logits and labels from the model's predictions.

    Returns:
        LuminarEvaluationMetrics: A dictionary containing calcualted metrics.
    """
    logits, labels = eval_pred  # type: ignore

    labels: NDArray = np.array(labels)
    scores: NDArray = 1 / (1 + np.exp(-np.array(logits)))

    n_samples_human = int(np.sum(labels == 0))
    n_samples_ai = int(np.sum(labels == 1))

    metrics = LuminarEvaluationMetrics(
        **compute_scores(scores, 0.5, labels),
        n_samples=len(labels),
    )

    if n_samples_human == n_samples_ai:
        # dataset is balanced, use the median of all scores as threshold
        threshold = float(np.median(scores))
        metrics_median = compute_scores(scores, threshold, labels, "_median")
        return Balanced(**metrics, **metrics_median, threshold_median=threshold)
    elif n_samples_human > 0 < n_samples_ai:
        # dataset is unbalanced, use the midpoint between the means of the two class distributions as threshold
        threshold = float(scores[labels == 0].mean() + scores[labels == 1].mean()) / 2
        metrics_mean = compute_scores(scores, threshold, labels, "_mean")
        return Unbalanced(
            **metrics,
            **metrics_mean,
            threshold_mean=threshold,
            n_samples_human=n_samples_human,
            n_samples_ai=n_samples_ai,
        )
    else:
        # only one class is present
        # TODO?
        pass

    return metrics
