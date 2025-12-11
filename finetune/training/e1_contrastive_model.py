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
    avg_sim_pos_pos: Optional[torch.FloatTensor] = None
    avg_sim_pos_neg: Optional[torch.FloatTensor] = None
    avg_sim_neg_pos: Optional[torch.FloatTensor] = None
    avg_sim_neg_neg: Optional[torch.FloatTensor] = None


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
        contrastive_label_mask: Optional[torch.BoolTensor] = None,
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

        # Process each view independently to avoid implicit inter-view coupling
        view_hidden_states = []
        mlm_losses = []

        for v in range(n_views):
            view_ids = input_ids[:, v, :]
            view_within = within_seq_position_ids[:, v, :]
            view_global = global_position_ids[:, v, :]
            view_seq_ids = sequence_ids[:, v, :]
            view_mlm_labels = mlm_labels[:, v, :] if mlm_labels is not None else None

            backbone_kwargs = dict(
                input_ids=view_ids,
                within_seq_position_ids=view_within,
                global_position_ids=view_global,
                sequence_ids=view_seq_ids,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            if view_mlm_labels is not None:
                backbone_kwargs["labels"] = view_mlm_labels

            backbone_outputs = self._original_model(**backbone_kwargs)
            view_hidden_states.append(backbone_outputs.last_hidden_state)

            if view_mlm_labels is not None:
                if (
                    hasattr(backbone_outputs, "loss")
                    and backbone_outputs.loss is not None
                ):
                    mlm_losses.append(backbone_outputs.loss)
                else:
                    mlm_logits = getattr(backbone_outputs, "logits", None)
                    if mlm_logits is None:
                        mlm_logits = self.mlm_head(backbone_outputs.last_hidden_state)
                    mlm_losses.append(
                        self.ce_loss(
                            mlm_logits.view(-1, mlm_logits.size(-1)),
                            view_mlm_labels.view(-1),
                        )
                    )

        # Stack hidden states back: [batch, n_views, seq_len, hidden]
        hidden_states = torch.stack(view_hidden_states, dim=1)

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
        avg_sim_pos_pos = None
        avg_sim_pos_neg = None
        avg_sim_neg_pos = None
        avg_sim_neg_neg = None

        # Compute contrastive + prototype + BCE loss if labels provided
        if binding_labels is not None and loss_fn is not None:
            # Prefer contrastive_label_mask (excludes masked tokens) if provided
            effective_mask = (
                contrastive_label_mask
                if contrastive_label_mask is not None
                else label_mask
            )
            if effective_mask is None:
                effective_mask = torch.ones(
                    batch_size, seq_len, dtype=torch.bool, device=device
                )

            # Extract valid residue embeddings per sample
            all_features = []
            all_labels = []

            if effective_mask.dim() == 2:
                # Shared mask across views
                for b in range(batch_size):
                    valid_pos = effective_mask[b]
                    num_valid = valid_pos.sum().item()
                    if num_valid == 0:
                        continue

                    sample_emb = hidden_states[b, :, valid_pos, :].transpose(0, 1)
                    sample_labels = binding_labels[b, valid_pos]
                    all_features.append(sample_emb)
                    all_labels.append(sample_labels)
            else:
                # Per-view masks – align residues by order within each view
                for b in range(batch_size):
                    per_view_embs = []
                    per_view_labels = []
                    min_valid = None

                    for v in range(n_views):
                        view_mask = effective_mask[b, v]
                        emb_v = hidden_states[b, v][view_mask]
                        if binding_labels.dim() == 3:
                            labels_v = binding_labels[b, v][view_mask]
                        else:
                            labels_v = binding_labels[b][view_mask]

                        valid_len = emb_v.size(0)
                        if valid_len == 0:
                            continue

                        min_valid = (
                            valid_len
                            if min_valid is None
                            else min(min_valid, valid_len)
                        )
                        per_view_embs.append(emb_v)
                        per_view_labels.append(labels_v)

                    if min_valid is None or min_valid == 0:
                        continue

                    # Truncate all views to the minimum valid length to align residues
                    aligned_embs = [emb[:min_valid] for emb in per_view_embs]
                    aligned_labels = per_view_labels[0][:min_valid]

                    sample_emb = torch.stack(aligned_embs, dim=0).transpose(0, 1)
                    all_features.append(sample_emb)
                    all_labels.append(aligned_labels)

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
                avg_sim_pos_pos = loss_dict.get("avg_sim_pos_pos")
                avg_sim_pos_neg = loss_dict.get("avg_sim_pos_neg")
                avg_sim_neg_pos = loss_dict.get("avg_sim_neg_pos")
                avg_sim_neg_neg = loss_dict.get("avg_sim_neg_neg")

                # Compute logits for metrics using average embedding across views
                avg_features = features.mean(dim=1)  # [N, hidden]

                from training.prototype_scoring import compute_prototype_distance_scores

                logits = compute_prototype_distance_scores(
                    embeddings=avg_features, prototypes=prototypes
                )

        # Compute MLM loss (average across views)
        if mlm_labels is not None and mlm_losses:
            loss_mlm = torch.stack(mlm_losses).mean()
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
            last_hidden_state=hidden_states,
            pull_mask_ratio=pull_mask_ratio,
            avg_sim_pos_pos=avg_sim_pos_pos,
            avg_sim_pos_neg=avg_sim_pos_neg,
            avg_sim_neg_pos=avg_sim_neg_pos,
            avg_sim_neg_neg=avg_sim_neg_neg,
        )
