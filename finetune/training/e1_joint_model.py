"""
Joint model for residue-level binding (BCE) + MLM auxiliary loss.

Wraps the existing E1 backbone with ion-specific heads and an MLM head.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from training.e1_classification_model import E1ForResidueClassification


class E1JointOutput(Dict[str, Any]):
    """Lightweight output container for joint training."""

    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(item)


class E1ForJointBindingMLM(E1ForResidueClassification):
    """
    E1 model with joint BCE (binding) and MLM losses.

    loss = loss_bce + mlm_weight * loss_mlm

    Supports configurable loss functions:
    - "bce": Binary Cross-Entropy with optional pos_weight
    - "focal": Focal Loss for handling class imbalance
    """

    def __init__(
        self,
        e1_model: nn.Module,
        ion_types,
        dropout: float = 0.1,
        mlm_weight: float = 0.4,
        freeze_backbone: bool = False,
        loss_type: str = "bce",
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        """
        Initialize joint binding + MLM model.

        Args:
            e1_model: E1 model with LoRA adapters
            ion_types: List of ion types for classification heads
            dropout: Dropout rate for classification heads
            mlm_weight: Weight for MLM loss (loss = BCE + mlm_weight * MLM)
            freeze_backbone: Whether to freeze backbone (not needed with LoRA)
            loss_type: "bce" or "focal"
            focal_gamma: Gamma parameter for focal loss
            focal_alpha: Alpha parameter for focal loss
        """
        if not hasattr(e1_model, "mlm_head"):
            raise ValueError(
                "Joint training requires a pretrained mlm_head on e1_model."
            )

        super().__init__(
            e1_model=e1_model,
            ion_types=ion_types,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
        )
        cfg = self.e1_backbone.config
        self.mlm_weight = mlm_weight

        # Use the pretrained MLM head from the backbone model
        self.mlm_head = e1_model.mlm_head

        pad_id = getattr(cfg, "pad_token_id", 0)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(
        self,
        input_ids: torch.LongTensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        ion: str,
        labels: Optional[torch.FloatTensor] = None,  # binding labels
        label_mask: Optional[torch.BoolTensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> E1JointOutput:
        # Backbone forward once; use the pretrained MLM model so we can reuse logits if available
        backbone_kwargs = dict(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        if mlm_labels is not None:
            backbone_kwargs["labels"] = mlm_labels

        backbone_outputs = self._original_model(**backbone_kwargs)
        hidden_states = backbone_outputs.last_hidden_state

        # Binding logits and loss
        # Binding logits and loss
        logits_binding = self.classifier_heads[ion](hidden_states)
        loss_bce = None
        if labels is not None:
            if self.loss_type == "focal":
                loss_bce = self.focal_loss_fn(logits_binding, labels.float())

            elif self.loss_type == "bce":

                if pos_weight is not None:
                    loss_bce = nn.functional.binary_cross_entropy_with_logits(
                        logits_binding,
                        labels.float(),
                        pos_weight=pos_weight,
                        reduction="none",
                    )
                else:
                    loss_bce = nn.functional.binary_cross_entropy_with_logits(
                        logits_binding, labels.float(), reduction="none"
                    )

            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            if label_mask is not None:
                loss_bce = loss_bce * label_mask.float()
                denom = label_mask.sum()
                loss_bce = loss_bce.sum() / (denom + 1e-8)
            else:
                loss_bce = loss_bce.mean()
        else:
            loss_bce = torch.tensor(0.0, device=input_ids.device)

        # MLM loss
        loss_mlm = None
        if mlm_labels is not None:
            if hasattr(backbone_outputs, "loss") and backbone_outputs.loss is not None:
                # Prefer loss computed by the pretrained MLM head
                loss_mlm = backbone_outputs.loss
            else:
                mlm_logits = getattr(backbone_outputs, "logits", None)
                if mlm_logits is None:
                    mlm_logits = self.mlm_head(hidden_states)
                # mlm_labels already have pad_token_id for ignore positions
                loss_mlm = self.ce_loss(
                    mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)
                )
        else:
            loss_mlm = torch.tensor(0.0, device=input_ids.device)

        total_loss = loss_bce + self.mlm_weight * loss_mlm

        return E1JointOutput(
            loss=total_loss,
            loss_bce=loss_bce,
            loss_mlm=loss_mlm,
            logits=logits_binding,
            last_hidden_state=hidden_states,
        )
