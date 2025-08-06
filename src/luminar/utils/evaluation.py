import numba as nb
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


@nb.jit(nopython=True)
def sign(x):
    return -1 if x < 0 else 1


type Threshold = float
type FPR = float


@nb.jit(nopython=True)
def find_threshold_for_fpr(
    y_scores: NDArray,
    target_fpr: float = 0.05,
    epsilon: float = 0.0005,
    less_than: bool = False,
) -> Threshold:
    """

    Modified from: RAID, Dugan et al. 2024.

    Source:
        https://github.com/liamdugan/raid/blob/main/raid/evaluate.py#L18

    Args:
        y_scores (list[float] | NDArray): The predicted scores for the human-written texts.
        target_fpr (float): The target false-positive rate to achieve.
        epsilon (float): The acceptable error margin for the false-positive rate.
        greater (bool, optional): If true, a score greater than the threshold is considered a positive prediction.

    Returns:
        tuple[Threshold, FPR]: a tuple of floats representing the threshold and the corresponding false-positive rate.
    """
    # Initialize the list of all found thresholds and FPRs
    prev_dist = np.nan
    step_size = 0.5
    found_threshold_list = []

    threshold: Threshold = y_scores.mean()
    for _ in range(50):
        # Calculate predictions as `greater-than-or-equal-to current threshold`
        # and flip them using XOR with `not greater` (i.e. the positive class is below the threshold)
        y_pred = (y_scores >= threshold) ^ less_than

        # Ground truth is 0 for human-written texts, so all predictions of 1 are false positives
        fpr = float(np.mean((y_pred) == 1))

        # If we reached the target FPR, return the threshold
        if abs(fpr - target_fpr) <= epsilon:
            return threshold

        # Save the computed values to the found_threshold_list
        found_threshold_list.append((threshold, fpr))

        # Compute distance
        dist = target_fpr - fpr

        # If dist and prev_dist are different signs then swap
        # sign of step size and cut in half
        if prev_dist != np.nan and sign(dist) != sign(prev_dist):
            step_size *= -0.5
        # Otherwise if we're going the wrong direction, then just swap sign of step
        elif prev_dist != np.nan and abs(dist) - abs(prev_dist) > 0.01:
            step_size *= -1

        # Step the threshold value and save prev_dist
        threshold += step_size
        prev_dist = dist

    # Compute diffs for all thresholds found during search
    # (Exclude all thresholds for which the true fpr is 0)
    diffs = [(target_fpr - fpr, t) for t, fpr in found_threshold_list if fpr > 0.0]

    # If there are positive numbers in the list, pick threshold for smallest pos number
    # Otherwise pick the threshold for the negative diff value closest to 0
    pos_diffs = [(d, t) for d, t in diffs if d >= 0]
    if len(pos_diffs) > 0:
        threshold = min(pos_diffs)[1]
    else:
        threshold = max(diffs)[1]

    return threshold


def calculate_metrics(
    y_true: NDArray,
    y_scores: NDArray,
    threshold: float,
    suffix="",
    less_than: bool = False,
    y_preds: NDArray | None = None,
) -> dict[str, float]:
    # Calculate predictions as `greater-than-or-equal-to current threshold`
    # and flip them using XOR with `less_than` (i.e. the positive class is below the threshold)
    if y_preds is None:
        y_preds = ((y_scores >= threshold) ^ less_than).astype(int)

    precision, recall, f1_score, _support = precision_recall_fscore_support(
        y_true, y_preds, average="weighted", zero_division=0
    )
    accuracy = np.mean(y_true == y_preds)
    fpr = np.mean(y_preds[y_true == 0] == 1)
    tpr = np.mean(y_preds[y_true == 1] == 1)

    if y_scores.min() < 0 or y_scores.max() > 1:
        norm_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    else:
        norm_scores = y_scores
    if less_than:
        norm_scores = 1 - norm_scores
    roc_auc = roc_auc_score(y_true, norm_scores, average="weighted")

    calculated_metrics = {
        f"f1_score{suffix}": float(f1_score),  # type: ignore
        f"precision{suffix}": float(precision),  # type: ignore
        f"recall{suffix}": float(recall),  # type: ignore
        f"accuracy{suffix}": float(accuracy),  # type: ignore
        f"roc_auc{suffix}": float(roc_auc),  # type: ignore
        f"fpr{suffix}": float(fpr),  # type: ignore
        f"tpr{suffix}": float(tpr),  # type: ignore
    }

    _precision_each, _recall_each, f1_score_each, _support_each = (
        precision_recall_fscore_support(y_true, y_preds, average=None, zero_division=0)
    )
    try:
        calculated_metrics |= {
            f"f1_human{suffix}": float(f1_score_each[0]),  # type: ignore
            f"f1_ai{suffix}": float(f1_score_each[1]),  # type: ignore
        }
    except Exception:
        UserWarning("precision_recall_fscore_support(average=None) raised an exception")

    return calculated_metrics


def run_evaluation(
    y_true: NDArray,
    y_scores: NDArray,
    threshold: float = 0.5,
    sigmoid: bool = True,
    less_than: bool = False,
    y_preds: NDArray | None = None,
) -> dict[str, float | int]:
    """Calculated weigthed and class-specific F1 scores, accuracy, ROC AUC, and class distribution for a threshold of 0.5.
    In addition, we compute best-guess thresholds based on the dataset balance:
    - If the dataset is balanced, we use the median of all scores as threshold.
    - If the dataset is unbalanced, we use the midpoint between the means of the two classes as threshold.
    - If only one class is present, no additional threshold is considered.

    Args:
        eval_pred: Tuple of logits and labels from the model's predictions.
        threshold: The threshold to use for classification.
        sigmoid: Whether to apply the sigmoid function to the logits.
        less_than: Whether a higher score indicates a positive class (False, default) or a lower score (True).

    Returns:
        EvaluationMetrics: A dictionary containing calcualted metrics.
    """
    if sigmoid:
        # Convert logits to probabilities using the sigmoid function
        y_scores = 1 / (1 + np.exp(-y_scores))

    n_samples_human = int(np.sum(y_true == 0))
    n_samples_ai = int(np.sum(y_true == 1))

    metrics = {"n_samples": len(y_true)}
    metrics |= calculate_metrics(
        y_true, y_scores, threshold, less_than=less_than, y_preds=y_preds
    )

    if n_samples_human == n_samples_ai:
        # dataset is balanced, use the median of all scores as threshold
        threshold_median = float(np.median(y_scores))
        metrics_median = calculate_metrics(
            y_true, y_scores, threshold_median, "_median", less_than=less_than
        )
        metrics |= metrics_median | {"threshold_median": threshold_median}

    # Use the midpoint between the means of the two class distributions as threshold
    # works if the dataset is unbalanced
    threshold_mean = (
        float(y_scores[y_true == 0].mean() + y_scores[y_true == 1].mean()) / 2
    )
    metrics_mean = calculate_metrics(
        y_true, y_scores, threshold_mean, "_mean", less_than=less_than
    )
    metrics |= metrics_mean | {"threshold_mean": threshold_mean}

    # Use the midpoint between the means of the two class distributions as threshold
    # works if the dataset is unbalanced
    threshold_fpr = find_threshold_for_fpr(
        y_scores[y_true == 0], target_fpr=0.05, epsilon=0.0005, less_than=less_than
    )
    metrics_fpr = calculate_metrics(
        y_true, y_scores, threshold_fpr, "_fpr", less_than=less_than
    )
    metrics |= metrics_fpr | {"threshold_fpr": threshold_fpr}

    if n_samples_human > 0 < n_samples_ai:
        metrics |= dict(
            n_samples_human=n_samples_human,
            n_samples_ai=n_samples_ai,
        )

    return metrics
