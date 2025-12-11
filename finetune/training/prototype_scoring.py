"""
Utility functions for prototype-based scoring and ranking.

This module provides common functions for computing ranking scores from prototype distances,
eliminating code duplication across the prototype ranking pipeline.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F


def compute_prototype_distance_scores(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    scoring_temperature: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    """
    Compute ranking scores from prototype distances using embeddings.

    This function handles both single-variant and multi-variant embeddings:
    - For multi-variant embeddings [batch_size, n_views, feat_dim]: computes scores for
      all variants and returns the average
    - For single-variant embeddings [batch_size, feat_dim]: computes scores directly

    The ranking score is computed as: similarity(x, prototype_pos) - similarity(x, prototype_neg)
    with optional temperature scaling applied to similarities before computing the difference.

    Args:
        embeddings: Input embeddings of shape [batch_size, feat_dim] or [batch_size, n_views, feat_dim]
        prototypes: Class prototypes of shape [2, feat_dim] where prototypes[0] is negative, prototypes[1] is positive
        scoring_temperature: Temperature for scaling similarities before computing difference (default: 1.0)
        logger: Optional logger for debugging and warnings

    Returns:
        Ranking scores of shape [batch_size]

    Raises:
        ValueError: If embeddings have unexpected dimensions or prototypes are invalid
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if prototypes is None:
        raise ValueError("Prototypes must be provided")

    if prototypes.shape[0] != 2:
        raise ValueError(
            f"Expected 2 prototypes (negative, positive), got {prototypes.shape[0]}"
        )

    try:
        # Handle multi-variant embeddings: [batch_size, n_views, feat_dim]
        if embeddings.dim() == 3:
            batch_size, n_views, feat_dim = embeddings.shape

            # Normalize all variants: [batch_size, n_views, feat_dim]
            normalized_embeddings = F.normalize(embeddings, p=2, dim=2)

            # Check for NaN/Inf in normalized embeddings
            if (
                torch.isnan(normalized_embeddings).any()
                or torch.isinf(normalized_embeddings).any()
            ):
                logger.warning("Normalized embeddings contain NaN or Inf values")
                return torch.zeros(
                    batch_size,
                    device=embeddings.device,
                    requires_grad=embeddings.requires_grad,
                )

            # Compute prototype similarities for all variants: [batch_size, n_views, 2]
            similarities = torch.matmul(normalized_embeddings, prototypes.T)

            # Apply temperature scaling to similarities before computing difference
            # This expands the dynamic range of the final scores
            scaled_similarities = similarities / scoring_temperature

            # Compute ranking scores for all variants: [batch_size, n_views]
            # Assumption: prototypes[1] is positive, prototypes[0] is negative
            distance_scores = (
                scaled_similarities[:, :, 1] - scaled_similarities[:, :, 0]
            )

            # Average ranking scores across variants: [batch_size]
            avg_distance_scores = distance_scores.mean(dim=1)

            return avg_distance_scores

        # Handle single-variant embeddings: [batch_size, feat_dim]
        elif embeddings.dim() == 2:
            batch_size, feat_dim = embeddings.shape

            # Normalize embeddings
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

            # Check for NaN/Inf in normalized embeddings
            if (
                torch.isnan(normalized_embeddings).any()
                or torch.isinf(normalized_embeddings).any()
            ):
                logger.warning("Normalized embeddings contain NaN or Inf values")
                return torch.zeros(
                    batch_size,
                    device=embeddings.device,
                    requires_grad=embeddings.requires_grad,
                )

            # Compute prototype similarities: [batch_size, 2]
            similarities = torch.matmul(normalized_embeddings, prototypes.T)

            # Apply temperature scaling to similarities before computing difference
            scaled_similarities = similarities / scoring_temperature

            # Compute ranking scores: [batch_size]
            # Assumption: prototypes[1] is positive, prototypes[0] is negative
            distance_scores = scaled_similarities[:, 1] - scaled_similarities[:, 0]

            return distance_scores

        else:
            raise ValueError(
                f"Expected embeddings to be 2D or 3D, got shape: {embeddings.shape}"
            )

    except Exception as e:
        logger.error(f"Error in prototype ranking scores computation: {str(e)}")
        batch_size = embeddings.shape[0]
        return torch.zeros(
            batch_size, device=embeddings.device, requires_grad=embeddings.requires_grad
        )


def compute_prototype_probabilities(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    scoring_temperature: float = 1.0,
    logger: Optional[logging.Logger] = None,
) -> torch.Tensor:
    """
    Compute classification probabilities from prototype distances.

    This is a convenience function that computes ranking scores and applies sigmoid
    to convert them to probabilities in the [0, 1] range.

    Args:
        embeddings: Input embeddings of shape [batch_size, feat_dim] or [batch_size, n_views, feat_dim]
        prototypes: Class prototypes of shape [2, feat_dim]
        scoring_temperature: Temperature for scaling similarities (default: 1.0)
        logger: Optional logger for debugging and warnings

    Returns:
        Classification probabilities of shape [batch_size]
    """
    distance_scores = compute_prototype_distance_scores(
        embeddings=embeddings,
        prototypes=prototypes,
        scoring_temperature=scoring_temperature,
        logger=logger,
    )

    # Apply sigmoid to get probabilities in [0, 1] range
    # This provides better calibration and handles the expanded range naturally
    probabilities = torch.sigmoid(distance_scores)

    return probabilities
