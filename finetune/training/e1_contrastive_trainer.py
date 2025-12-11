"""
Trainer for E1 contrastive learning with prototype alignment.

Extends E1BindingTrainer to handle:
1. Prototype initialization from class means (positive samples → positive prototype)
2. Multi-variant batch processing
3. Contrastive loss computation
"""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from training.e1_binding_trainer import E1BindingTrainer
from training.metrics import high_recall_auprc

logger = logging.getLogger(__name__)


class E1ContrastiveTrainer(E1BindingTrainer):
    """
    Trainer for contrastive learning with prototype alignment.

    Prototype Strategy:
        - Compute positive prototype as mean of positive residue embeddings
        - Negative prototype is automatically derived as: neg = -pos
        - Only update positive prototype via EMA during training
    """

    def __init__(
        self,
        model: nn.Module,
        conf: Any,
        device: torch.device,
        loss_fn: nn.Module,  # PrototypeBCELoss instance
        logger_instance: Optional[Any] = None,
        is_distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        """
        Initialize contrastive trainer.

        Args:
            model: E1ForContrastiveBinding model
            conf: Configuration object
            device: Training device
            loss_fn: PrototypeBCELoss instance
            logger_instance: Logger instance
            is_distributed: Whether using DDP
            world_size: Number of processes
            rank: Current process rank
        """
        super().__init__(
            model=model,
            conf=conf,
            device=device,
            logger_instance=logger_instance,
            is_distributed=is_distributed,
            world_size=world_size,
            rank=rank,
        )

        self.loss_fn = loss_fn

        if self.is_main_process:
            self.logger.info("E1ContrastiveTrainer initialized")
            self.logger.info(
                "  - Prototype init: class_mean (pos_prototype = mean of positive samples)"
            )

    def initialize_prototypes(
        self, train_loader: DataLoader, ion: str, max_samples: int = 100
    ):
        """
        Initialize positive prototype from mean of positive residue embeddings.

        Strategy:
            1. Collect residue embeddings from training data
            2. Compute mean of POSITIVE class embeddings → pos_prototype
            3. Model automatically derives neg_prototype = -pos_prototype

        Args:
            train_loader: Training data loader
            ion: Ion type
            max_samples: Maximum samples to use for computing mean
        """
        if self.is_main_process:
            self.logger.info(
                f"Initializing prototype for {ion} from positive class mean..."
            )

        self.model.eval()

        pos_embeddings = []
        n_samples = 0

        with torch.no_grad():
            for batch in train_loader:
                if n_samples >= max_samples:
                    break

                batch = self._move_batch_to_device(batch)

                # Get hidden states from backbone
                # For initialization, we only need single view (no masking)
                input_ids = batch["input_ids"]
                if input_ids.dim() == 3:  # [batch, n_views, seq_len]
                    input_ids = input_ids[:, 0, :]  # Take first view
                    within_pos = batch["within_seq_position_ids"][:, 0, :]
                    global_pos = batch["global_position_ids"][:, 0, :]
                    seq_ids = batch["sequence_ids"][:, 0, :]
                else:
                    within_pos = batch["within_seq_position_ids"]
                    global_pos = batch["global_position_ids"]
                    seq_ids = batch["sequence_ids"]

                with autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16 and torch.cuda.is_available(),
                ):
                    hidden = self._unwrapped_model.get_encoder_output(
                        input_ids=input_ids,
                        within_seq_position_ids=within_pos,
                        global_position_ids=global_pos,
                        sequence_ids=seq_ids,
                    )

                # Extract embeddings for valid residues
                label_mask = batch["label_mask"]
                if label_mask.dim() == 3:  # [batch, n_views, seq_len]
                    label_mask = label_mask[:, 0, :]

                labels = batch["binding_labels"]

                for b in range(hidden.shape[0]):
                    valid = label_mask[b]
                    valid_hidden = hidden[b, valid, :].float().cpu()
                    valid_labels = labels[b, valid]

                    # Only collect positive class embeddings
                    pos_mask = valid_labels == 1
                    if pos_mask.any():
                        pos_embeddings.append(valid_hidden[pos_mask.cpu()])

                n_samples += hidden.shape[0]

        if not pos_embeddings:
            if self.is_main_process:
                self.logger.error(
                    f"No positive samples found for {ion}! Cannot initialize prototype - model will not train properly."
                )
            return

        # Compute positive class mean
        pos_mean = torch.cat(pos_embeddings, dim=0).mean(dim=0)

        # Set on model (model will normalize internally)
        self._unwrapped_model.set_positive_prototype(ion, pos_mean.to(self.device))

        if self.is_main_process:
            total_pos = sum(e.shape[0] for e in pos_embeddings)
            self.logger.info(
                f"Initialized {ion} positive prototype from {total_pos} positive residues "
                f"(neg_prototype = -pos_prototype)"
            )

        self.model.train()

    def train_epoch(
        self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, ion: str
    ) -> Dict[str, float]:
        """
        Train for one epoch with contrastive learning.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            ion: Ion type being trained

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_loss_contrastive = 0.0
        total_loss_prototype = 0.0
        total_loss_bce = 0.0
        total_loss_mlm = 0.0
        all_probs = []
        all_labels = []
        num_batches = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)

            with autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.use_bf16 and torch.cuda.is_available(),
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    within_seq_position_ids=batch["within_seq_position_ids"],
                    global_position_ids=batch["global_position_ids"],
                    sequence_ids=batch["sequence_ids"],
                    ion=ion,
                    binding_labels=batch["binding_labels"],
                    label_mask=batch["label_mask"],
                    mlm_labels=batch.get("mlm_labels"),
                    loss_fn=self.loss_fn,
                )

                loss = outputs.loss / self.accum_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()
            if outputs.loss_contrastive is not None:
                total_loss_contrastive += outputs.loss_contrastive.item()
            if outputs.loss_prototype is not None:
                total_loss_prototype += outputs.loss_prototype.item()
            if outputs.loss_bce is not None:
                total_loss_bce += outputs.loss_bce.item()
            if outputs.loss_mlm is not None:
                total_loss_mlm += outputs.loss_mlm.item()
            num_batches += 1

            # Collect predictions for metrics
            if outputs.logits is not None:
                with torch.no_grad():
                    probs = torch.sigmoid(outputs.logits)
                    label_mask = batch["label_mask"]
                    labels = batch["binding_labels"]

                    # Flatten valid labels to match logits
                    valid_labels = []
                    for b in range(labels.shape[0]):
                        valid_labels.append(labels[b, label_mask[b]])
                    flat_labels = torch.cat(valid_labels)

                    all_probs.append(probs.detach().cpu())
                    all_labels.append(flat_labels.detach().cpu())

        # Handle remaining gradients
        if num_batches % self.accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Compute metrics
        train_metrics = {
            "loss": total_loss / max(num_batches, 1),
            "loss_contrastive": total_loss_contrastive / max(num_batches, 1),
            "loss_prototype": total_loss_prototype / max(num_batches, 1),
            "loss_bce": total_loss_bce / max(num_batches, 1),
            "loss_mlm": total_loss_mlm / max(num_batches, 1),
        }

        if all_probs:
            all_probs = torch.cat(all_probs).float()
            all_labels = (torch.cat(all_labels) > 0.5).long()

            train_metrics["auc"] = self.metric_auc(all_probs, all_labels).item()
            train_metrics["auprc"] = self.metric_auprc(all_probs, all_labels).item()
            train_metrics["high_recall_auprc"] = high_recall_auprc(
                all_labels, all_probs, recall_threshold=0.7
            )
        else:
            train_metrics["auc"] = 0.0
            train_metrics["auprc"] = 0.0
            train_metrics["high_recall_auprc"] = 0.0

        # Track metrics
        self.metrics_tracker.update_train(train_metrics, ion)

        return train_metrics

    @torch.no_grad()
    def validate_epoch(
        self, val_loader: DataLoader, ion: str, fold_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate for one epoch using prototype distance scores.

        Args:
            val_loader: Validation data loader
            ion: Ion type being validated
            fold_idx: Fold index for OOF predictions

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_loss_bce = 0.0
        total_loss_mlm = 0.0
        all_probs = []
        all_labels = []
        num_batches = 0

        collecting_oof = self.is_main_process and fold_idx is not None

        for batch in val_loader:
            batch = self._move_batch_to_device(batch)

            with autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.use_bf16 and torch.cuda.is_available(),
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    within_seq_position_ids=batch["within_seq_position_ids"],
                    global_position_ids=batch["global_position_ids"],
                    sequence_ids=batch["sequence_ids"],
                    ion=ion,
                    binding_labels=batch["binding_labels"],
                    label_mask=batch["label_mask"],
                    mlm_labels=batch.get("mlm_labels"),
                    loss_fn=self.loss_fn,
                )

            if outputs.loss is not None:
                total_loss += outputs.loss.item()
            if outputs.loss_bce is not None:
                total_loss_bce += outputs.loss_bce.item()
            if outputs.loss_mlm is not None:
                total_loss_mlm += outputs.loss_mlm.item()
            num_batches += 1

            # Collect predictions
            if outputs.logits is not None:
                probs = torch.sigmoid(outputs.logits)
                label_mask = batch["label_mask"]
                labels = batch["binding_labels"]

                valid_labels = []
                for b in range(labels.shape[0]):
                    valid_labels.append(labels[b, label_mask[b]])
                flat_labels = torch.cat(valid_labels)

                all_probs.append(probs.cpu())
                all_labels.append(flat_labels.cpu())

                # OOF collection
                if collecting_oof:
                    self._accumulate_oof_from_batch(
                        ion=ion,
                        probs=probs.unsqueeze(0) if probs.dim() == 1 else probs,
                        labels=labels,
                        label_mask=label_mask,
                        protein_ids=batch.get("protein_ids"),
                        fold_idx=fold_idx,
                    )

        # Compute metrics
        all_probs = torch.cat(all_probs).float() if all_probs else torch.tensor([])
        all_labels = torch.cat(all_labels).long() if all_labels else torch.tensor([])

        if len(all_probs) > 0:
            val_auc = self.metric_auc(all_probs, all_labels).item()
            val_auprc = self.metric_auprc(all_probs, all_labels).item()
            val_hra = high_recall_auprc(all_labels, all_probs, recall_threshold=0.7)

            from training.metrics import find_optimal_threshold

            threshold_method = getattr(self.conf.training, "threshold_method", "youden")
            threshold_results = find_optimal_threshold(
                all_probs, all_labels, method=threshold_method
            )
        else:
            val_auc = val_auprc = val_hra = 0.0
            threshold_results = {
                "f1": 0.0,
                "mcc": 0.0,
                "threshold": 0.5,
                "recall": 0.0,
                "precision": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
            }

        val_metrics = {
            "loss": total_loss / max(num_batches, 1),
            "loss_bce": total_loss_bce / max(num_batches, 1),
            "loss_mlm": total_loss_mlm / max(num_batches, 1),
            "auc": val_auc,
            "auprc": val_auprc,
            "high_recall_auprc": val_hra,
            "f1": threshold_results["f1"],
            "mcc": threshold_results["mcc"],
            "threshold": threshold_results["threshold"],
            "recall": threshold_results["recall"],
            "precision": threshold_results["precision"],
            "confusion_matrix": threshold_results["confusion_matrix"],
        }

        self.metrics_tracker.update_val(val_metrics, ion)

        return val_metrics
