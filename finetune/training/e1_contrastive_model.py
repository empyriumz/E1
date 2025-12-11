"""
E1 model for residue-level contrastive learning with prototype alignment.

This model extends E1ForJointBindingMLM to support:
1. Multi-variant forward passes (num_variants masked views)
2. Single positive prototype per ion with neg_prototype = -pos_prototype
3. Residue-level contrastive + prototype + BCE loss computation
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from training.e1_joint_model import E1ForJointBindingMLM
from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class E1ContrastiveOutput(ModelOutput):
    """Output class for E1 contrastive learning model."""

    loss: Optional[torch.FloatTensor] = None
    loss_contrastive: Optional[torch.FloatTensor] = None
    loss_prototype: Optional[torch.FloatTensor] = None
    loss_bce: Optional[torch.FloatTensor] = None
    loss_mlm: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None  # BCE logits from prototype scores
    embeddings: Optional[torch.FloatTensor] = None  # [batch, n_views, seq_len, hidden]
    last_hidden_state: Optional[torch.FloatTensor] = None
    # Diagnostic metrics
    pull_mask_ratio: Optional[torch.FloatTensor] = None
    avg_sim_pos: Optional[torch.FloatTensor] = None
    avg_sim_neg: Optional[torch.FloatTensor] = None


class E1ForContrastiveBinding(E1ForJointBindingMLM):
    """
    E1 model for residue-level contrastive learning with prototype alignment.

    Prototype Strategy:
        - Only the POSITIVE prototype is learned per ion type
        - Negative prototype is derived as: neg_prototype = -pos_prototype
        - This constraint enforces symmetric separation in embedding space

    Architecture:
        Input [batch, n_views, seq_len]
        → Flatten to [batch*n_views, seq_len]
        → E1 backbone → hidden states [batch*n_views, seq_len, hidden]
        → Reshape to [batch, n_views, seq_len, hidden]
        → Residue-level losses on valid positions

    Note: LoRA handles backbone freezing - no separate freeze_backbone parameter.
    """

    def __init__(
        self,
        e1_model: nn.Module,
        ion_types: List[str],
        dropout: float = 0.1,
        # Prototype configuration
        prototype_dim: Optional[int] = None,  # Defaults to hidden_size
        # MLM configuration
        mlm_weight: float = 1.0,
        # Legacy parameters (ignored - kept for backward compatibility)
        use_ema_prototypes: bool = True,  # Ignored - prototypes are now frozen
        ema_decay: float = 0.999,  # Ignored - prototypes are now frozen
    ):
        """
        Initialize E1 contrastive learning model.

        Args:
            e1_model: E1 model with LoRA adapters (backbone frozen via LoRA)
            ion_types: List of ion types for classification heads
            dropout: Dropout rate for classification heads
            prototype_dim: Dimension of prototypes (defaults to hidden_size)
            mlm_weight: Weight for MLM loss component
            use_ema_prototypes: IGNORED - prototypes are now frozen after init
            ema_decay: IGNORED - prototypes are now frozen after init
        """
        # Initialize parent with BCE loss only (no focal loss)
        super().__init__(
            e1_model=e1_model,
            ion_types=ion_types,
            dropout=dropout,
            mlm_weight=mlm_weight,
            freeze_backbone=False,  # LoRA handles this
            loss_type="bce",  # Only BCE supported
        )

        # Prototype configuration
        self.prototype_dim = prototype_dim or self.hidden_size
        self.ion_types = ion_types

        # Frozen positive prototypes (negative = -positive)
        # Registered as buffers (not learnable) - initialized from class means in trainer
        for ion in ion_types:
            self.register_buffer(
                f"pos_prototype_{ion}", torch.zeros(self.prototype_dim)
            )

        # Track if prototypes have been initialized
        self._prototypes_initialized = {ion: False for ion in ion_types}

        logger.info("E1ForContrastiveBinding initialized:")
        logger.info(f"  - Prototype dim: {self.prototype_dim}")
        logger.info(f"  - Ion types: {ion_types}")
        logger.info("  - Prototype strategy: FROZEN, neg_prototype = -pos_prototype")

    def get_prototypes(self, ion: str) -> torch.Tensor:
        """
        Get both prototypes for an ion: [neg_prototype, pos_prototype].

        Returns:
            Tensor of shape [2, hidden_size] where:
                prototypes[0] = -pos_prototype (negative class)
                prototypes[1] = pos_prototype (positive class)
        """
        pos_proto = getattr(self, f"pos_prototype_{ion}")

        # Normalize the positive prototype
        pos_proto_norm = F.normalize(pos_proto.unsqueeze(0), p=2, dim=1).squeeze(0)

        # Negative prototype is the negation of positive
        neg_proto_norm = -pos_proto_norm

        # Stack: [neg, pos]
        return torch.stack([neg_proto_norm, pos_proto_norm], dim=0)

    def set_positive_prototype(self, ion: str, prototype: torch.Tensor):
        """
        Set the positive prototype for an ion type (frozen after this).

        Args:
            ion: Ion type
            prototype: Prototype vector of shape [hidden_size]
        """
        with torch.no_grad():
            # Normalize and set
            proto_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1).squeeze(0)
            buffer = getattr(self, f"pos_prototype_{ion}")
            buffer.copy_(proto_norm)
            self._prototypes_initialized[ion] = True

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        ion: str,
        binding_labels: Optional[torch.FloatTensor] = None,
        label_mask: Optional[torch.BoolTensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        loss_fn: Optional[nn.Module] = None,  # PrototypeBCELoss instance
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> E1ContrastiveOutput:
        """
        Forward pass for contrastive learning.

        Args:
            input_ids: [batch, n_views, seq_len] token IDs (with masking applied)
            within_seq_position_ids: [batch, n_views, seq_len]
            global_position_ids: [batch, n_views, seq_len]
            sequence_ids: [batch, n_views, seq_len]
            ion: Ion type for classification
            binding_labels: [batch, seq_len] residue-level binary labels
            label_mask: [batch, seq_len] valid residue positions (query sequence)
            mlm_labels: [batch, n_views, seq_len] MLM targets
            loss_fn: PrototypeBCELoss instance for contrastive losses
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights

        Returns:
            E1ContrastiveOutput with loss components and embeddings
        """
        if ion not in self.ion_types:
            raise ValueError(f"Unknown ion type: {ion}")

        if not self._prototypes_initialized[ion]:
            logger.warning(f"Prototype for {ion} not initialized - using zeros")

        # Get input dimensions
        batch_size, n_views, seq_len = input_ids.shape
        device = input_ids.device

        # Flatten for backbone: [batch*n_views, seq_len]
        flat_input_ids = input_ids.view(batch_size * n_views, seq_len)
        flat_within_pos = within_seq_position_ids.view(batch_size * n_views, seq_len)
        flat_global_pos = global_position_ids.view(batch_size * n_views, seq_len)
        flat_seq_ids = sequence_ids.view(batch_size * n_views, seq_len)

        # Flatten MLM labels if provided
        flat_mlm_labels = None
        if mlm_labels is not None:
            flat_mlm_labels = mlm_labels.view(batch_size * n_views, seq_len)

        # Forward through backbone
        backbone_kwargs = dict(
            input_ids=flat_input_ids,
            within_seq_position_ids=flat_within_pos,
            global_position_ids=flat_global_pos,
            sequence_ids=flat_seq_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        if flat_mlm_labels is not None:
            backbone_kwargs["labels"] = flat_mlm_labels

        backbone_outputs = self._original_model(**backbone_kwargs)
        flat_hidden = (
            backbone_outputs.last_hidden_state
        )  # [batch*n_views, seq_len, hidden]

        # Reshape hidden states: [batch, n_views, seq_len, hidden]
        hidden_states = flat_hidden.view(batch_size, n_views, seq_len, -1)

        # Get prototypes for this ion [2, hidden]
        prototypes = self.get_prototypes(ion)

        # Initialize loss components
        loss_total = torch.tensor(0.0, device=device, requires_grad=True)
        loss_contrastive = None
        loss_prototype = None
        loss_bce = None
        loss_mlm = None
        logits = None
        # Diagnostic metrics
        pull_mask_ratio = None
        avg_sim_pos = None
        avg_sim_neg = None

        # Compute contrastive + prototype + BCE loss if labels provided
        if binding_labels is not None and loss_fn is not None:
            # Use label_mask to identify valid query residues
            # All views share the same valid positions (masking doesn't exclude from loss)
            if label_mask is not None:
                valid_mask = label_mask  # [batch, seq_len]
            else:
                valid_mask = torch.ones(
                    batch_size, seq_len, dtype=torch.bool, device=device
                )

            # Extract valid residue embeddings per sample
            # Each residue gets n_views embeddings for contrastive learning
            all_features = []
            all_labels = []

            for b in range(batch_size):
                valid_pos = valid_mask[b]  # [seq_len]
                num_valid = valid_pos.sum().item()

                if num_valid == 0:
                    continue

                # Extract embeddings for valid residues: [n_views, num_valid, hidden]
                sample_emb = hidden_states[b, :, valid_pos, :]

                # Transpose to [num_valid, n_views, hidden]
                sample_emb = sample_emb.transpose(0, 1)

                # Get labels for valid residues: [num_valid]
                sample_labels = binding_labels[b, valid_pos]

                all_features.append(sample_emb)
                all_labels.append(sample_labels)

            if all_features:
                # Concatenate: [total_residues, n_views, hidden]
                features = torch.cat(all_features, dim=0)
                labels = torch.cat(all_labels, dim=0)

                # Set prototypes on loss function
                loss_fn.set_prototypes(prototypes)

                # Compute loss using PrototypeBCELoss
                loss_dict = loss_fn(features, labels, training=self.training)

                loss_contrastive = loss_dict.get("contrastive")
                loss_prototype = loss_dict.get("prototype")
                loss_bce = loss_dict.get("bce")
                loss_total = loss_dict["total"]
                # Extract diagnostic metrics
                pull_mask_ratio = loss_dict.get("pull_mask_ratio")
                avg_sim_pos = loss_dict.get("avg_sim_pos")
                avg_sim_neg = loss_dict.get("avg_sim_neg")

                # Compute logits for metrics using average embedding across views
                avg_features = features.mean(dim=1)  # [N, hidden]

                from training.prototype_scoring import compute_prototype_distance_scores

                logits = compute_prototype_distance_scores(
                    embeddings=avg_features, prototypes=prototypes
                )

        # Compute MLM loss (average across views)
        if mlm_labels is not None:
            if hasattr(backbone_outputs, "loss") and backbone_outputs.loss is not None:
                loss_mlm = backbone_outputs.loss
            else:
                mlm_logits = getattr(backbone_outputs, "logits", None)
                if mlm_logits is None:
                    mlm_logits = self.mlm_head(flat_hidden)
                loss_mlm = self.ce_loss(
                    mlm_logits.view(-1, mlm_logits.size(-1)), flat_mlm_labels.view(-1)
                )

            # Add weighted MLM loss to total
            if loss_mlm is not None:
                loss_total = loss_total + self.mlm_weight * loss_mlm

        # Prototypes are frozen - no EMA update needed

        return E1ContrastiveOutput(
            loss=loss_total,
            loss_contrastive=loss_contrastive,
            loss_prototype=loss_prototype,
            loss_bce=loss_bce,
            loss_mlm=loss_mlm,
            logits=logits,
            embeddings=hidden_states,
            last_hidden_state=flat_hidden.view(batch_size, n_views, seq_len, -1),
            pull_mask_ratio=pull_mask_ratio,
            avg_sim_pos=avg_sim_pos,
            avg_sim_neg=avg_sim_neg,
        )
