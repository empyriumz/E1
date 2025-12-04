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
    ):
        """
        Initialize E1ForResidueClassification.

        Args:
            e1_model: Pre-loaded E1 model (E1ForMaskedLM with LoRA or E1Model)
            ion_types: List of ion types to create classification heads for
            dropout: Dropout rate for classification heads
            freeze_backbone: Whether to freeze the E1 backbone (only train heads)
        """
        super().__init__(e1_model.config)

        self.ion_types = ion_types or self.DEFAULT_ION_TYPES

        if not hasattr(e1_model, "mlm_head"):
            raise ValueError(
                "e1_model must provide an mlm_head (expected E1ForMaskedLM checkpoint)."
            )

        # Reuse backbone and MLM head from provided model
        self.e1_backbone = e1_model.model if hasattr(e1_model, "model") else e1_model
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

    def to(self, device=None, dtype=None, *args, **kwargs):
        """Move model to device and/or dtype."""
        self._original_model.to(device=device, dtype=dtype, *args, **kwargs)
        self.classifier_heads.to(device=device, dtype=dtype, *args, **kwargs)
        return self

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
            pos_weight: Positive class weight for BCE loss
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
            # Create loss function
            if pos_weight is not None:
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight, reduction="none"
                )
            else:
                criterion = nn.BCEWithLogitsLoss(reduction="none")

            # Compute loss
            loss = criterion(logits, labels.float())

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
