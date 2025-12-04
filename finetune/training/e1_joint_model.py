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
    """

    def __init__(
        self,
        e1_model: nn.Module,
        ion_types,
        dropout: float = 0.1,
        mlm_weight: float = 0.4,
        freeze_backbone: bool = False,
    ):
        if not hasattr(e1_model, "mlm_head"):
            raise ValueError(
                "Joint training requires a pretrained mlm_head on e1_model."
            )

        super().__init__(
            e1_model=e1_model,
            ion_types=ion_types,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
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
        # Backbone forward once
        hidden_states = self.get_encoder_output(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Binding logits and loss
        logits_binding = self.classifier_heads[ion](hidden_states)
        loss_bce = None
        if labels is not None:
            if pos_weight is not None:
                criterion = nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight, reduction="none"
                )
            else:
                criterion = nn.BCEWithLogitsLoss(reduction="none")
            loss_bce = criterion(logits_binding, labels.float())
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
