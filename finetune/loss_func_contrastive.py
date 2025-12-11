from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.prototype_scoring import compute_prototype_distance_scores


class ConSupPrototypeLoss(nn.Module):
    """
    Improved ConSup loss with prototype alignment and unsupervised contrastive learning.

    Key improvements:
    1. Consistent normalization across all similarity computations
    2. Simplified margin conditions with single eps parameter
    3. Enhanced numerical stability
    4. Better gradient flow management
    """

    def __init__(
        self,
        temperature: float = 0.07,
        eps: float = 0.1,
        prototype_weight: float = 1.0,
        unsupervised_weight: float = 1.0,  # Renamed from contrastive_weight
        device: Optional[torch.device] = None,
        logger=None,
        eps_pos: Optional[float] = None,
        eps_neg: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the improved ConSup Prototype loss function.

        Args:
            temperature: Temperature scaling parameter for similarity computations.
            eps: Margin parameter for prototype distance (single value for both classes).
            prototype_weight: Weight for prototype alignment loss component.
            unsupervised_weight: Weight for unsupervised contrastive learning loss component.
            device: The device to run the computations on.
            logger: Logger instance to use.
            eps_pos: Asymmetric margin for the positive class (class 1). Falls back to `eps`.
            eps_neg: Asymmetric margin for the negative class (class 0). Falls back to `eps`.
        """
        super().__init__()
        self.temperature = temperature
        self.prototype_weight = prototype_weight
        self.unsupervised_weight = (
            unsupervised_weight  # Renamed from contrastive_weight
        )

        self.device = device

        # Use provided logger or create a default one
        if logger is not None:
            self.logger = logger
        else:
            import logging

            self.logger = logging.getLogger(__name__)

        # Use asymmetric eps if provided, otherwise fall back to symmetric eps
        self.eps_pos = eps_pos if eps_pos is not None else eps
        self.eps_neg = eps_neg if eps_neg is not None else eps

        self.logger.info(
            f"Initialized ConSupPrototypeLoss with eps_pos={self.eps_pos:.4f}, eps_neg={self.eps_neg:.4f}"
        )

        self.prototypes = None

        # For numerical stability - increased from 1e-8 to 1e-6 for better stability
        self.eps_numerical = 1e-6

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """Set the class prototypes to use in the loss calculation."""
        try:
            # Move prototypes to the correct device
            self.prototypes = prototypes.to(self.device)

            # Validate prototypes
            if torch.isnan(self.prototypes).any() or torch.isinf(self.prototypes).any():
                raise ValueError("Prototypes contain NaN or Inf values")

        except Exception as e:
            self.logger.error(f"Failed to set prototypes: {str(e)}")
            raise

    def _compute_unsupervised_contrastive_loss(self, features, batch_size, n_views):
        """
        Compute unsupervised contrastive loss with improved numerical stability.
        """
        if n_views < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        try:
            # Reshape and normalize features consistently
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

            # Check for NaN/Inf in features
            if (
                torch.isnan(contrast_feature).any()
                or torch.isinf(contrast_feature).any()
            ):
                self.logger.warning(
                    "Features contain NaN or Inf values, returning zero loss"
                )
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Compute similarity matrix with temperature scaling
            sim_matrix = (
                torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
            )

            # For numerical stability
            sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            logits = sim_matrix - sim_matrix_max.detach()

            # Create positive mask (same original sequence, different views)
            labels = torch.arange(batch_size).repeat(n_views).to(self.device)
            mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()

            # Mask out self-contrast cases
            self_mask = torch.eye(
                batch_size * n_views, device=self.device, dtype=torch.bool
            )
            mask = mask * (~self_mask).float()

            # Check for valid positive pairs
            num_positives_per_sample = mask.sum(1)
            if (num_positives_per_sample == 0).all():
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Compute log probabilities with numerical stability
            exp_logits = torch.exp(logits)
            exp_logits = exp_logits * (~self_mask).float()  # Mask out self-similarities

            log_prob = logits - torch.log(
                exp_logits.sum(1, keepdim=True) + self.eps_numerical
            )

            # Compute loss only for samples with valid positive pairs
            valid_samples = num_positives_per_sample > 0
            if not valid_samples.any():
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            mean_log_prob_pos = (mask * log_prob).sum(
                1
            ) / num_positives_per_sample.clamp(min=self.eps_numerical)
            valid_mean_log_prob_pos = mean_log_prob_pos[valid_samples]

            # Simplified loss computation (no base_temperature scaling)
            loss = -valid_mean_log_prob_pos.mean()

            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    "Contrastive loss is NaN or Inf, returning zero loss"
                )
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            return loss

        except Exception as e:
            self.logger.error(f"Error in contrastive loss computation: {str(e)}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _compute_prototype_alignment_loss(self, features, labels, batch_size, n_views):
        """
        Compute prototype alignment loss with improved stability.
        """
        if self.prototypes is None:
            raise ValueError("Prototypes must be set before computing alignment loss.")

        try:
            # Reshape and normalize features consistently
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            contrast_feature = F.normalize(contrast_feature, p=2, dim=1)

            # Check for NaN/Inf in features
            if (
                torch.isnan(contrast_feature).any()
                or torch.isinf(contrast_feature).any()
            ):
                self.logger.warning(
                    "Features contain NaN or Inf values, returning zero loss"
                )
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            # Compute normalized similarities to prototypes
            sim_to_prototypes = torch.matmul(contrast_feature, self.prototypes.T)
            sim_to_p0 = sim_to_prototypes[:, 0]
            sim_to_p1 = sim_to_prototypes[:, 1]

            # Get class indices for each sample (repeated for multiple views)
            # Handle both 1D and 2D label formats robustly
            if labels.dim() == 1:
                # Labels are already class indices (0 or 1)
                class_indices = labels
                # Validate that labels contain only valid class indices
                if not torch.all((class_indices == 0) | (class_indices == 1)):
                    raise ValueError(
                        f"Labels must contain only 0 or 1, got: {torch.unique(class_indices)}"
                    )
            elif labels.dim() == 2:
                # Labels are one-hot encoded [batch_size, num_classes]
                if labels.shape[1] != 2:
                    raise ValueError(
                        f"Expected 2 classes in one-hot labels, got shape: {labels.shape}"
                    )
                class_indices = labels.argmax(dim=1)
            else:
                raise ValueError(
                    f"Unexpected labels dimension: {labels.dim()}, shape: {labels.shape}"
                )

            if class_indices.dim() > 1:
                class_indices = class_indices.squeeze()

            if n_views == 1:
                repeated_labels = class_indices
            else:
                repeated_labels = class_indices.repeat(n_views)

            is_class_0 = repeated_labels == 0
            is_class_1 = repeated_labels == 1

            # Pull samples that are not sufficiently separated from wrong prototype
            pull_mask_0 = is_class_0 & (sim_to_p0 <= sim_to_p1 + self.eps_neg)
            pull_mask_1 = is_class_1 & (sim_to_p1 <= sim_to_p0 + self.eps_pos)
            pull_mask = (pull_mask_0 | pull_mask_1).float()

            # Compute pull mask ratio for diagnostics
            pull_mask_ratio = pull_mask.sum() / pull_mask.numel()

            # Compute prototype alignment loss with temperature scaling
            proto_logits = sim_to_prototypes / self.temperature
            log_softmax_logits = F.log_softmax(proto_logits, dim=1)

            # Negative log likelihood for correct prototype
            nll_loss = -log_softmax_logits[
                torch.arange(batch_size * n_views), repeated_labels.long()
            ]

            # Apply conditional pulling mask
            num_pulled = pull_mask.sum().clamp(min=self.eps_numerical)
            if num_pulled == 0:
                return (
                    torch.tensor(0.0, device=self.device, requires_grad=True),
                    pull_mask_ratio,
                )

            loss = (nll_loss * pull_mask).sum() / num_pulled

            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(
                    "Prototype alignment loss is NaN or Inf, returning zero loss"
                )
                return (
                    torch.tensor(0.0, device=self.device, requires_grad=True),
                    pull_mask_ratio,
                )

            # Also compute average similarities to prototypes for diagnostics
            avg_sim_pos = sim_to_p1.mean()
            avg_sim_neg = sim_to_p0.mean()

            return loss, {
                "pull_mask_ratio": pull_mask_ratio.detach(),
                "avg_sim_pos": avg_sim_pos.detach(),
                "avg_sim_neg": avg_sim_neg.detach(),
            }

        except Exception as e:
            self.logger.error(
                f"Error in prototype alignment loss computation: {str(e)}"
            )
            import traceback

            self.logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=self.device, requires_grad=True), {
                "pull_mask_ratio": torch.tensor(0.0),
                "avg_sim_pos": torch.tensor(0.0),
                "avg_sim_neg": torch.tensor(0.0),
            }

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """
        Compute the improved ConSup loss.

        Args:
            features: Tensor of shape [bsz, n_views, feat_dim] containing the embeddings.
            labels: Tensor of shape [bsz, 2] with one-hot class encodings.
            training: Whether this is a training step (for compatibility, not used in ConSup loss).

        Returns:
            Dictionary containing individual loss components and total loss.
        """
        if labels is None:
            raise ValueError("Labels must be provided for ConSup loss.")

        if self.device is None:
            self.device = features.device

        try:
            batch_size, n_views, _ = features.shape
            loss_components = {}
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Unsupervised contrastive learning component (if multiple views available)
            if n_views > 1 and self.unsupervised_weight > 0:
                contrastive_loss = self._compute_unsupervised_contrastive_loss(
                    features, batch_size, n_views
                )
                weighted_contrastive = self.unsupervised_weight * contrastive_loss
                total_loss = total_loss + weighted_contrastive
                loss_components["contrastive"] = contrastive_loss.detach()

            # Prototype alignment component
            if self.prototype_weight > 0:
                prototype_loss, proto_diagnostics = (
                    self._compute_prototype_alignment_loss(
                        features, labels, batch_size, n_views
                    )
                )
                weighted_prototype = self.prototype_weight * prototype_loss
                total_loss = total_loss + weighted_prototype
                loss_components["prototype"] = prototype_loss.detach()
                # Include diagnostic metrics
                loss_components["pull_mask_ratio"] = proto_diagnostics[
                    "pull_mask_ratio"
                ]
                loss_components["avg_sim_pos"] = proto_diagnostics["avg_sim_pos"]
                loss_components["avg_sim_neg"] = proto_diagnostics["avg_sim_neg"]

            # Add total loss to components
            loss_components["total"] = total_loss

            # Final check for NaN/Inf in total loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.error(
                    "Total loss is NaN or Inf, returning zero loss components"
                )
                return {
                    "total": torch.tensor(0.0, device=self.device, requires_grad=True),
                    "contrastive": torch.tensor(0.0, device=self.device),
                    "prototype": torch.tensor(0.0, device=self.device),
                }

            return loss_components

        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            return {
                "total": torch.tensor(0.0, device=self.device, requires_grad=True),
                "contrastive": torch.tensor(0.0, device=self.device),
                "prototype": torch.tensor(0.0, device=self.device),
            }


class PrototypeBCELoss(ConSupPrototypeLoss):
    """
    Prototype-based BCE loss that combines prototype alignment with binary classification.

    The key insight: embeddings must be aligned to meaningful prototypes before
    prototype distances can be used for effective classification.

    This loss optimizes for:
    1. Prototype alignment: Aligns embeddings to learned class prototypes (ESSENTIAL)
    2. Classification performance: Uses aligned prototype distances for BCE classification
    3. Optional contrastive: Additional regularization across variants

    Design principle: Learn prototypes → Align embeddings → Classify by distances
    """

    def __init__(
        self,
        bce_weight: float = 1.0,  # Weight for BCE loss component
        unsupervised_weight: float = 0.5,  # Weight for unsupervised contrastive component
        prototype_weight: float = 1.0,  # MUST enable prototype alignment for meaningful classification
        scoring_temperature: float = 1.0,  # Temperature for final scoring (separate from contrastive temperature)
        label_smoothing: float = 0.0,  # Label smoothing factor (0.0 = no smoothing)
        **kwargs,  # All ConSupPrototypeLoss parameters
    ):
        """
        Initialize PrototypeBCELoss with classification capabilities.

        Args:
            bce_weight: Weight for BCE loss component (set to 0 to disable)
            unsupervised_weight: Weight for unsupervised contrastive component (set to 0 to disable)
            prototype_weight: Weight for prototype alignment (MUST be > 0 for meaningful classification)
            scoring_temperature: Temperature for final scoring (separate from contrastive temperature)
            **kwargs: All parameters from ConSupPrototypeLoss
        """
        # Initialize parent with weight-based control
        super().__init__(
            unsupervised_weight=unsupervised_weight,
            prototype_weight=prototype_weight,
            **kwargs,
        )

        self.bce_weight = bce_weight
        self.scoring_temperature = scoring_temperature
        self.label_smoothing = label_smoothing

        if bce_weight > 0:
            self.logger.info(
                f"Extended ConSupPrototypeLoss for classification: bce_weight={bce_weight}, unsupervised_weight={unsupervised_weight}, scoring_temperature={scoring_temperature}, label_smoothing={label_smoothing}"
            )
        else:
            self.logger.info(
                f"Extended ConSupPrototypeLoss for classification: DISABLED (weight=0), unsupervised_weight={unsupervised_weight}, scoring_temperature={scoring_temperature}, label_smoothing={label_smoothing}"
            )

    def _compute_classification_scores(self, features: torch.Tensor) -> torch.Tensor:
        """Compute classification scores from prototype distances using all variants."""
        if self.prototypes is None:
            raise ValueError("Prototypes must be set")

        return compute_prototype_distance_scores(
            embeddings=features,
            prototypes=self.prototypes,
            scoring_temperature=self.scoring_temperature,
            logger=self.logger,
        )

    def _compute_bce_loss(
        self, scores: torch.Tensor, labels: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        """
        Binary cross-entropy with logits on prototype-difference scores.

        Treats the similarity difference s = sim(x, p_pos) - sim(x, p_neg) as the logit
        for class 1. This directly optimizes calibrated classification consistent with
        downstream sigmoid(thresholding).

        Applies label smoothing if label_smoothing > 0 and training=True:
        - Label 1 becomes (1 - label_smoothing)
        - Label 0 becomes label_smoothing
        """
        try:
            # Ensure 1D tensors
            if scores.dim() > 1:
                scores = scores.squeeze()
            if labels.dim() > 1:
                labels = labels.squeeze()

            targets = labels.float()

            # Apply label smoothing only during training
            if training and self.label_smoothing > 0:
                # For binary classification:
                # Label 1 -> (1 - label_smoothing)
                # Label 0 -> label_smoothing
                targets = (
                    targets * (1.0 - self.label_smoothing)
                    + (1.0 - targets) * self.label_smoothing
                )

            loss = F.binary_cross_entropy_with_logits(scores, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning("BCE loss is NaN/Inf; returning zero")
                return torch.tensor(0.0, device=scores.device, requires_grad=True)

            return loss
        except Exception as e:
            self.logger.error(f"Error in BCE loss computation: {str(e)}")
            return torch.tensor(0.0, device=scores.device, requires_grad=True)

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> dict:
        """
        Compute prototype BCE loss.

        Args:
            features: Tensor of shape [batch_size, n_views, feat_dim]
            labels: Binary labels
            training: Whether this is a training step (affects label smoothing)

        Returns:
            Dictionary with loss components
        """
        if labels is None:
            raise ValueError("Labels required for classification loss")

        if self.device is None:
            self.device = features.device

        try:
            batch_size, n_views, _ = features.shape
            loss_components = {}
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Log variant information for debugging
            if hasattr(self, "logger") and hasattr(self.logger, "debug"):
                self.logger.debug(
                    f"PrototypeBCELoss: batch_size={batch_size}, n_views={n_views}, unsupervised_weight={self.unsupervised_weight}"
                )

            # 1. Prototype alignment component (ESSENTIAL for meaningful classification)
            if self.prototype_weight > 0:
                prototype_loss, proto_diagnostics = (
                    self._compute_prototype_alignment_loss(
                        features, labels, batch_size, n_views
                    )
                )
                weighted_prototype = self.prototype_weight * prototype_loss
                total_loss = total_loss + weighted_prototype
                loss_components["prototype"] = prototype_loss.detach()
                # Include diagnostic metrics
                loss_components["pull_mask_ratio"] = proto_diagnostics[
                    "pull_mask_ratio"
                ]
                loss_components["avg_sim_pos"] = proto_diagnostics["avg_sim_pos"]
                loss_components["avg_sim_neg"] = proto_diagnostics["avg_sim_neg"]

            # 2. Optional contrastive component (if multiple views available)
            if self.unsupervised_weight > 0 and n_views > 1:
                contrastive_loss = self._compute_unsupervised_contrastive_loss(
                    features, batch_size, n_views
                )
                weighted_contrastive = self.unsupervised_weight * contrastive_loss
                total_loss = total_loss + weighted_contrastive
                loss_components["contrastive"] = contrastive_loss.detach()

            # 3. BCE classification component using prototype distances (skip if weight is 0)
            if self.bce_weight > 0:
                # Convert labels to class indices for BCE loss computation
                if labels.dim() == 2:
                    class_indices = labels.argmax(dim=1)
                elif labels.dim() > 1:
                    class_indices = labels.squeeze()
                else:
                    class_indices = labels

                # Compute classification scores and BCE loss
                classification_scores = self._compute_classification_scores(features)

                # Monitor classification scores gradients
                if not classification_scores.requires_grad:
                    if hasattr(self, "logger") and hasattr(self.logger, "debug"):
                        self.logger.debug(
                            "Classification scores do not require gradients (possibly during validation)"
                        )

                bce_loss = self._compute_bce_loss(
                    classification_scores, class_indices, training=training
                )

                # Monitor BCE loss gradients
                if not bce_loss.requires_grad:
                    if hasattr(self, "logger") and hasattr(self.logger, "debug"):
                        self.logger.debug(
                            "BCE loss does not require gradients (possibly during validation)"
                        )

                # Apply BCE weight and add to total loss
                weighted_bce_loss = self.bce_weight * bce_loss
                loss_components["bce"] = bce_loss.detach()
                total_loss = total_loss + weighted_bce_loss
            else:
                # Skip BCE loss computation when weight is 0
                loss_components["bce"] = torch.tensor(0.0, device=self.device)

            loss_components["total"] = total_loss

            # Final check for NaN/Inf in total loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self.logger.error(
                    "Total prototype BCE loss is NaN or Inf, returning zero loss components"
                )
                return {
                    "total": torch.tensor(0.0, device=self.device, requires_grad=True),
                    "prototype": torch.tensor(0.0, device=self.device),
                    "contrastive": torch.tensor(0.0, device=self.device),
                    "bce": torch.tensor(0.0, device=self.device),
                }

            return loss_components

        except Exception as e:
            self.logger.error(f"Error in prototype BCE loss forward pass: {str(e)}")
            return {
                "total": torch.tensor(0.0, device=self.device, requires_grad=True),
                "prototype": torch.tensor(0.0, device=self.device),
                "contrastive": torch.tensor(0.0, device=self.device),
                "bce": torch.tensor(0.0, device=self.device),
            }
