"""
E1 model for residue-level metal ion binding classification.

This module provides E1ForResidueClassification, which adds ion-specific
classification heads on top of the E1 backbone for per-residue binary
classification of metal binding sites.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

from transformers.modeling_outputs import ModelOutput
from modeling_e1 import E1ForMaskedLM

logger = logging.getLogger(__name__)


@dataclass
class E1ResidueClassificationOutput(ModelOutput):
    """Output class for E1 residue classification model."""

    loss: Optional[torch.FloatTensor] = None
    loss_bce: Optional[torch.FloatTensor] = None
    loss_mlm: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class IonSpecificClassifierHead(nn.Module):
    """Classification head for a specific ion type."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # Binary classification per residue
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            logits: (batch_size, seq_len)
        """
        return self.classifier(hidden_states).squeeze(-1)


class E1ForResidueClassification(E1ForMaskedLM):
    """
    E1 model with ion-specific classification heads for residue-level binding prediction.

    This model wraps an E1 backbone (with LoRA adapters) and adds ion-specific
    classification heads similar to ESMBindSingle's architecture.

    Architecture:
        E1 Backbone (LoRA) -> Ion-Specific Heads -> Per-Residue Logits
    """

    # Default ion types matching ESMBindSingle
    DEFAULT_ION_TYPES = ["CA", "ZN", "MG", "LREE", "HREE", "REE"]

    def __init__(
        self,
        e1_model: nn.Module,
        ion_types: Optional[List[str]] = None,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        """
        Initialize E1ForResidueClassification.

        Args:
            e1_model: Pre-loaded E1 model (E1ForMaskedLM with LoRA or E1Model)
            ion_types: List of ion types to create classification heads for
            dropout: Dropout rate for classification heads
            freeze_backbone: Whether to freeze the E1 backbone (only train heads)
            loss_type: "bce" or "focal"
            focal_gamma: Gamma parameter for focal loss
            focal_alpha: Alpha parameter for focal loss
        """
        super().__init__(e1_model.config)

        self.ion_types = ion_types or self.DEFAULT_ION_TYPES

        if not hasattr(e1_model, "mlm_head"):
            raise ValueError(
                "e1_model must provide an mlm_head (expected E1ForMaskedLM checkpoint)."
            )

        # Cleanup unused randomly initialized model from super().__init__
        # This prevents it from sitting on CPU and causing DDP errors
        if hasattr(self, "model"):
            del self.model

        # Reuse backbone and MLM head from provided model
        self.e1_backbone = e1_model.model if hasattr(e1_model, "model") else e1_model

        # Alias self.model to backbone for compatibility with inherited methods (like _embed)
        # Note: Since e1_backbone is part of _original_model (PeftModel), this is an alias to a GPU-resident module
        self.model = self.e1_backbone

        self._original_model = e1_model
        self.mlm_head = e1_model.mlm_head

        # Hidden size from config (already set by super)
        self.hidden_size = self.e1_backbone.config.hidden_size

        # Create ion-specific classification heads
        self.classifier_heads = nn.ModuleDict(
            {
                ion: IonSpecificClassifierHead(self.hidden_size, dropout)
                for ion in self.ion_types
            }
        )

        # Initialize classification heads
        self._init_classifier_weights()

        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()

        logger.info(f"E1ForResidueClassification initialized:")
        logger.info(f"  - Hidden size: {self.hidden_size}")
        logger.info(f"  - Ion types: {self.ion_types}")
        logger.info(f"  - Dropout: {dropout}")
        logger.info(f"  - Backbone frozen: {freeze_backbone}")
        logger.info(f"  - Loss type: {loss_type}")

        # Store loss configuration and pre-create loss function
        self.loss_type = loss_type
        # Store loss configuration and pre-create loss function
        self.loss_type = loss_type
        self.focal_loss_fn = None

        if loss_type == "focal":
            from training.loss_functions import FocalLoss

            self.focal_loss_fn = FocalLoss(
                gamma=focal_gamma, alpha=focal_alpha, reduction="none"
            )
            logger.info(f"  - Focal loss: gamma={focal_gamma}, alpha={focal_alpha}")

    def _init_classifier_weights(self):
        """Initialize classification head weights with Xavier uniform."""
        for head in self.classifier_heads.values():
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _freeze_backbone(self):
        """Freeze the E1 backbone parameters."""
        for param in self.e1_backbone.parameters():
            param.requires_grad = False
        logger.info("E1 backbone parameters frozen")

    # NOTE: Removed custom to() method. The default nn.Module.to() implementation
    # correctly recurses into all submodules (including _original_model and classifier_heads).
    # The previous manual override failed to move self.model (if it existed) or other attributes.

    def get_encoder_output(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Get hidden states from E1 backbone.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            within_seq_position_ids: Position IDs within each sequence
            global_position_ids: Global position IDs
            sequence_ids: Sequence IDs for multi-sequence inputs

        Returns:
            last_hidden_state: (batch_size, seq_len, hidden_size)
        """
        outputs = self.e1_backbone(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        ion: str,
        labels: Optional[torch.FloatTensor] = None,
        label_mask: Optional[torch.BoolTensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> E1ResidueClassificationOutput:
        """
        Forward pass for residue-level classification.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            within_seq_position_ids: Position IDs within each sequence
            global_position_ids: Global position IDs
            sequence_ids: Sequence IDs for multi-sequence inputs
            ion: Ion type to classify (e.g., "CA", "ZN")
            labels: Binary labels (batch_size, seq_len), values: 0, 1, or ignore_index
            label_mask: Boolean mask for valid label positions (batch_size, seq_len)
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights

        Returns:
            E1ResidueClassificationOutput with loss, logits, and hidden states
        """
        if ion not in self.classifier_heads:
            raise ValueError(
                f"Unknown ion type: {ion}. Available: {list(self.classifier_heads.keys())}"
            )

        # Get hidden states from backbone
        hidden_states = self.get_encoder_output(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Get logits from ion-specific head
        logits = self.classifier_heads[ion](hidden_states)  # (batch_size, seq_len)

        loss = None
        if labels is not None:
            if self.loss_type == "focal":
                loss = self.focal_loss_fn(logits, labels.float())

            elif self.loss_type == "bce":

                if pos_weight is not None:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, labels.float(), pos_weight=pos_weight, reduction="none"
                    )
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        logits, labels.float(), reduction="none"
                    )

            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            # Apply mask if provided
            if label_mask is not None:
                loss = loss * label_mask.float()
                # Mean over valid positions only
                num_valid = label_mask.sum()
                if num_valid > 0:
                    loss = loss.sum() / num_valid
                else:
                    loss = loss.sum() * 0.0  # No valid positions, return 0 loss
            else:
                loss = loss.mean()

        return E1ResidueClassificationOutput(
            loss=loss,
            loss_bce=loss,
            loss_mlm=None,
            logits=logits,
            last_hidden_state=hidden_states,
        )

    def get_logits(self, hidden_states: torch.Tensor, ion: str) -> torch.Tensor:
        """
        Get classification logits for a specific ion type.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            ion: Ion type

        Returns:
            logits: (batch_size, seq_len)
        """
        if ion not in self.classifier_heads:
            raise ValueError(f"Unknown ion type: {ion}")
        return self.classifier_heads[ion](hidden_states)

    def save_classifier_heads(self, path: str):
        """Save only the classification heads (not backbone)."""
        torch.save(
            {
                "classifier_heads": self.classifier_heads.state_dict(),
                "ion_types": self.ion_types,
                "hidden_size": self.hidden_size,
            },
            path,
        )
        logger.info(f"Saved classifier heads to {path}")

    def load_classifier_heads(self, path: str):
        """Load classification heads from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.classifier_heads.load_state_dict(checkpoint["classifier_heads"])
        logger.info(f"Loaded classifier heads from {path}")

    @classmethod
    def from_pretrained_e1(
        cls,
        e1_model: nn.Module,
        ion_types: Optional[List[str]] = None,
        dropout: float = 0.1,
        classifier_checkpoint: Optional[str] = None,
    ) -> "E1ForResidueClassification":
        """
        Create E1ForResidueClassification from a pre-trained E1 model.

        Args:
            e1_model: Pre-loaded E1 model (with LoRA if desired)
            ion_types: List of ion types
            dropout: Dropout rate
            classifier_checkpoint: Optional path to pre-trained classifier heads

        Returns:
            E1ForResidueClassification instance
        """
        model = cls(
            e1_model=e1_model,
            ion_types=ion_types,
            dropout=dropout,
        )

        if classifier_checkpoint is not None:
            model.load_classifier_heads(classifier_checkpoint)

        return model

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        # Also set backbone to train mode
        self.e1_backbone.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def parameters(self, recurse: bool = True):
        """Return all parameters including backbone and heads."""
        yield from self._original_model.parameters(recurse)
        yield from self.classifier_heads.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Return named parameters."""
        for name, param in self._original_model.named_parameters(
            prefix=f"{prefix}backbone." if prefix else "backbone.", recurse=recurse
        ):
            yield name, param

        for name, param in self.classifier_heads.named_parameters(
            prefix=f"{prefix}classifier_heads." if prefix else "classifier_heads.",
            recurse=recurse,
        ):
            yield name, param
