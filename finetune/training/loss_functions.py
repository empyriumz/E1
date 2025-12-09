"""
Loss functions for E1 residue-level binding classification.

This module provides:
- FocalLoss: Focal loss for handling class imbalance
- compute_pos_weight: Utility to compute pos_weight from class counts
- get_loss_function: Factory to get loss function from config
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

PosWeightMode = Literal["linear", "sqrt"] | None


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.

    Focal Loss = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
    - p_t = p if y=1 else 1-p (probability of correct class)
    - α_t = α if y=1 else 1-α (class balancing factor)
    - γ = focusing parameter (higher = more focus on hard examples)

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    Args:
        gamma: Focusing parameter. Higher values put more weight on hard examples.
               Default: 2.0
        alpha: Positive class weight. Set to 0.25 for detection tasks, or higher
               for highly imbalanced problems. Default: 0.25
        reduction: 'none', 'mean', or 'sum'. Default: 'none' to allow masking.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "none",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits (before sigmoid), shape: (batch, seq_len) or (N,)
            targets: Binary targets (0 or 1), same shape as logits. Supports soft labels.

        Returns:
            Loss tensor with shape depending on reduction mode
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute α_t (class weight)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal weight: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma

        # Compute BCE loss (more numerically stable than manual log)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Combine: focal_loss = α_t * focal_weight * BCE
        # Note: This is an approximation. The exact Focal Loss formula is -α_t (1-p_t)^γ log(p_t)
        # BCE gives -[y log p + (1-y) log (1-p)]
        # This modification effectively applies the focal term to the BCE gradients.
        focal_loss = alpha_t * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # "none"
            return focal_loss


def compute_pos_weight(
    pos_count: int,
    neg_count: int,
    mode: PosWeightMode = None,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """
    Compute pos_weight for BCEWithLogitsLoss based on class counts.

    Args:
        pos_count: Number of positive samples
        neg_count: Number of negative samples
        mode: Weighting mode:
            - None: Return None (no weighting, standard BCE)
            - "linear": pos_weight = neg_count / pos_count
            - "sqrt": pos_weight = sqrt(neg_count / pos_count)
        device: Device to create tensor on

    Returns:
        pos_weight tensor or None if mode is None
    """
    if mode is None:
        logger.info("pos_weight_mode: None (no class weighting)")
        return None

    if pos_count == 0:
        logger.warning("No positive samples found, returning pos_weight=1.0")
        return torch.tensor([1.0], device=device)

    ratio = neg_count / pos_count

    if mode == "linear":
        pos_weight = ratio
        logger.info(
            f"pos_weight_mode: linear, pos_weight={pos_weight:.2f} "
            f"(pos={pos_count}, neg={neg_count}, ratio={ratio:.1f}:1)"
        )
    elif mode == "sqrt":
        pos_weight = math.sqrt(ratio)
        logger.info(
            f"pos_weight_mode: sqrt, pos_weight={pos_weight:.2f} "
            f"(pos={pos_count}, neg={neg_count}, ratio={ratio:.1f}:1, sqrt={pos_weight:.2f})"
        )
    else:
        raise ValueError(
            f"Unknown pos_weight_mode: {mode}. Use None, 'linear', or 'sqrt'"
        )

    return torch.tensor([pos_weight], dtype=torch.float, device=device)


def get_loss_function(
    loss_type: str = "bce",
    pos_weight: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
) -> nn.Module:
    """
    Factory function to create loss function from config.

    Args:
        loss_type: "bce" or "focal"
        pos_weight: Positive class weight (only used for BCE)
        focal_gamma: Gamma parameter for focal loss
        focal_alpha: Alpha parameter for focal loss

    Returns:
        Loss function module with reduction="none"
    """
    if loss_type == "bce":
        if pos_weight is not None:
            logger.info(
                f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.2f}"
            )
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        else:
            logger.info("Using BCEWithLogitsLoss (no pos_weight)")
            return nn.BCEWithLogitsLoss(reduction="none")

    elif loss_type == "focal":
        logger.info(f"Using FocalLoss with gamma={focal_gamma}, alpha={focal_alpha}")
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha, reduction="none")

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'bce' or 'focal'")
