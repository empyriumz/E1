import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve
import logging

logger = logging.getLogger(__name__)


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
