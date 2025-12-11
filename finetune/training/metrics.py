import logging
from typing import Any, Dict, List

import numpy as np
import torch
from torchmetrics.classification import (
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinaryROC,
)

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks training and validation metrics per ion for plotting."""

    def __init__(self):
        self.train_history: Dict[str, List[Dict]] = {}
        self.val_history: Dict[str, List[Dict]] = {}

    def update_train(self, metrics: Dict, ion: str):
        if ion not in self.train_history:
            self.train_history[ion] = []
        self.train_history[ion].append(metrics.copy())

    def update_val(self, metrics: Dict, ion: str):
        if ion not in self.val_history:
            self.val_history[ion] = []
        self.val_history[ion].append(metrics.copy())

    def get_all_ions(self) -> List[str]:
        return list(set(self.train_history.keys()) | set(self.val_history.keys()))

    def get_ion_metrics(self, ion: str) -> Dict:
        return {
            "train_metrics": self.train_history.get(ion, []),
            "val_metrics": self.val_history.get(ion, []),
        }


def find_optimal_threshold(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    method: str = "youden",
) -> Dict[str, Any]:
    """
    Find optimal classification threshold.

    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        method: 'youden', 'f1', or 'mcc'

    Returns:
        Dictionary with threshold and metrics
    """
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # Convert to float32 for metric computation
    predictions = predictions.float()
    labels = labels.long()

    if method == "youden":
        roc = BinaryROC(thresholds=None)
        fpr, tpr, thresholds = roc(predictions, labels)
        fpr = fpr.numpy()
        tpr = tpr.numpy()
        thresholds = thresholds.float().numpy()
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = float(thresholds[optimal_idx])

    elif method == "f1":
        thresholds = np.linspace(0.01, 0.99, 100)
        f1_scores = []
        for t in thresholds:
            f1_metric = BinaryF1Score(threshold=t)
            f1_scores.append(f1_metric(predictions, labels).item())
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[optimal_idx])

    elif method == "mcc":
        thresholds = np.linspace(0.01, 0.99, 100)
        mcc_scores = []
        for t in thresholds:
            mcc_metric = BinaryMatthewsCorrCoef(threshold=t)
            mcc_scores.append(mcc_metric(predictions, labels).item())
        optimal_idx = np.argmax(mcc_scores)
        optimal_threshold = float(thresholds[optimal_idx])

    else:
        optimal_threshold = 0.5

    # Calculate metrics at optimal threshold
    f1_metric = BinaryF1Score(threshold=optimal_threshold)
    mcc_metric = BinaryMatthewsCorrCoef(threshold=optimal_threshold)
    recall_metric = BinaryRecall(threshold=optimal_threshold)
    precision_metric = BinaryPrecision(threshold=optimal_threshold)
    cm_metric = BinaryConfusionMatrix(threshold=optimal_threshold)

    return {
        "threshold": optimal_threshold,
        "f1": f1_metric(predictions, labels).item(),
        "mcc": mcc_metric(predictions, labels).item(),
        "recall": recall_metric(predictions, labels).item(),
        "precision": precision_metric(predictions, labels).item(),
        "confusion_matrix": cm_metric(predictions, labels).numpy(),
    }


def high_recall_auprc(y_true, y_pred_proba, recall_threshold=0.7):
    """
    Calculate partial AUPRC for recall values above a specified threshold.

    This metric focuses on the high-recall region of the precision-recall curve,
    which is often more relevant for applications where missing positive cases is costly.

    Args:
        y_true (torch.Tensor or array-like): Ground truth labels
        y_pred_proba (torch.Tensor or array-like): Predicted probabilities
        recall_threshold (float): Minimum recall threshold (default: 0.7)

    Returns:
        float: Partial AUPRC for recall >= recall_threshold
    """
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred_proba, torch.Tensor):
        y_pred_proba = torch.tensor(y_pred_proba)

    # Use torchmetrics BinaryPrecisionRecallCurve
    pr_curve = BinaryPrecisionRecallCurve(thresholds=None)
    precision, recall, thresholds = pr_curve(y_pred_proba, y_true)

    # Convert to float32 if needed for numerical stability
    precision = precision.float()
    recall = recall.float()

    # Find indices where recall >= threshold
    high_recall_mask = recall >= recall_threshold

    if not torch.any(high_recall_mask):
        # If no recall values meet the threshold, return 0
        logger.warning(
            f"No recall values >= {recall_threshold}. Returning 0 for high_recall_auprc."
        )
        return 0.0

    # Get precision and recall values in the high-recall region
    high_recall_precision = precision[high_recall_mask]
    high_recall_recall = recall[high_recall_mask]

    # Sort by recall for proper integration
    sort_idx = torch.argsort(high_recall_recall)
    high_recall_precision = high_recall_precision[sort_idx]
    high_recall_recall = high_recall_recall[sort_idx]

    # Calculate partial AUPRC using trapezoidal integration
    if len(high_recall_recall) == 1:
        # Single point - no area to integrate
        return 0.0

    # Use PyTorch's trapezoidal integration
    partial_auprc = torch.trapz(high_recall_precision, high_recall_recall)

    return partial_auprc.item()
