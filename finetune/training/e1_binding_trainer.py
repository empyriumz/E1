"""
Trainer for E1 residue-level metal ion binding classification.

This module provides E1BindingTrainer which handles:
- Training with BCE loss and class balancing
- Validation with metrics computation (AUPRC, F1, MCC)
- OOF predictions storage for global threshold optimization
- Distributed Data Parallel (DDP) support for multi-GPU training
"""

import torch
import torch.nn as nn
import numpy as np
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryRecall,
    BinaryPrecision,
    BinaryConfusionMatrix,
    BinaryROC,
)
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks training and validation metrics per ion for plotting."""

    def __init__(self):
        self.train_history: Dict[str, List[Dict]] = {}
        self.val_history: Dict[str, List[Dict]] = {}

    def update_train(self, metrics: Dict, ion: str):
        if ion not in self.train_history:
            self.train_history[ion] = []
        self.train_history[ion].append(metrics.copy())

    def update_val(self, metrics: Dict, ion: str):
        if ion not in self.val_history:
            self.val_history[ion] = []
        self.val_history[ion].append(metrics.copy())

    def get_all_ions(self) -> List[str]:
        return list(set(self.train_history.keys()) | set(self.val_history.keys()))

    def get_ion_metrics(self, ion: str) -> Dict:
        return {
            "train_metrics": self.train_history.get(ion, []),
            "val_metrics": self.val_history.get(ion, []),
        }


class EarlyStopper:
    """Early stopping handler."""

    def __init__(self, patience: int = 7, warmup_epochs: int = 3, mode: str = "max"):
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.current_epoch = 0

    def early_stop(self, value: float) -> bool:
        self.current_epoch += 1

        # Don't trigger during warmup
        if self.current_epoch <= self.warmup_epochs:
            if self.mode == "max":
                self.best_value = max(self.best_value, value)
            else:
                self.best_value = min(self.best_value, value)
            return False

        improved = (
            value > self.best_value if self.mode == "max" else value < self.best_value
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


def find_optimal_threshold(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    method: str = "youden",
) -> Dict[str, Any]:
    """
    Find optimal classification threshold.

    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        method: 'youden', 'f1', or 'mcc'

    Returns:
        Dictionary with threshold and metrics
    """
    if predictions.ndim > 1:
        predictions = predictions.flatten()

    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # Convert to float32 for metric computation
    predictions = predictions.float()
    labels = labels.long()

    if method == "youden":
        roc = BinaryROC(thresholds=None)
        fpr, tpr, thresholds = roc(predictions, labels)
        fpr = fpr.numpy()
        tpr = tpr.numpy()
        thresholds = thresholds.float().numpy()
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = float(thresholds[optimal_idx])

    elif method == "f1":
        thresholds = np.linspace(0.01, 0.99, 100)
        f1_scores = []
        for t in thresholds:
            f1_metric = BinaryF1Score(threshold=t)
            f1_scores.append(f1_metric(predictions, labels).item())
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[optimal_idx])

    elif method == "mcc":
        thresholds = np.linspace(0.01, 0.99, 100)
        mcc_scores = []
        for t in thresholds:
            mcc_metric = BinaryMatthewsCorrCoef(threshold=t)
            mcc_scores.append(mcc_metric(predictions, labels).item())
        optimal_idx = np.argmax(mcc_scores)
        optimal_threshold = float(thresholds[optimal_idx])

    else:
        optimal_threshold = 0.5

    # Calculate metrics at optimal threshold
    f1_metric = BinaryF1Score(threshold=optimal_threshold)
    mcc_metric = BinaryMatthewsCorrCoef(threshold=optimal_threshold)
    recall_metric = BinaryRecall(threshold=optimal_threshold)
    precision_metric = BinaryPrecision(threshold=optimal_threshold)
    cm_metric = BinaryConfusionMatrix(threshold=optimal_threshold)

    return {
        "threshold": optimal_threshold,
        "f1": f1_metric(predictions, labels).item(),
        "mcc": mcc_metric(predictions, labels).item(),
        "recall": recall_metric(predictions, labels).item(),
        "precision": precision_metric(predictions, labels).item(),
        "confusion_matrix": cm_metric(predictions, labels).numpy(),
    }


class E1BindingTrainer:
    """
    Trainer for E1 residue-level binding classification.

    Handles training and validation with:
    - BCE loss with pos_weight for class imbalance
    - Masked loss computation (only valid residue positions)
    - Metrics tracking and early stopping
    - Distributed Data Parallel (DDP) support
    """

    def __init__(
        self,
        model: nn.Module,
        conf: Any,
        device: torch.device,
        logger_instance: Optional[Any] = None,
        is_distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        """
        Initialize trainer.

        Args:
            model: E1ForResidueClassification model (or DDP-wrapped model)
            conf: Configuration object
            device: Device to run training on
            logger_instance: Logger instance
            is_distributed: Whether using distributed training
            world_size: Number of processes in distributed training
            rank: Current process rank
        """
        self.model = model
        self.conf = conf
        self.device = device
        self.logger = logger_instance or logger
        self.is_distributed = is_distributed
        self.world_size = world_size
        self.rank = rank
        self.is_main_process = rank == 0

        # Get the underlying model if wrapped in DDP
        self._unwrapped_model = model.module if hasattr(model, "module") else model

        # Note: Model should already be on device from load_e1_for_classification
        # We just verify it's on the correct device
        if hasattr(self._unwrapped_model, "e1_backbone"):
            # Check first parameter's device
            first_param = next(self._unwrapped_model.e1_backbone.parameters(), None)
            if first_param is not None and first_param.device != device:
                self.logger.warning(f"Model not on expected device {device}, moving...")
                self.model.to(device)

        # Mixed precision settings
        self.use_bf16 = getattr(conf.training, "use_bf16", True)

        # Initialize metrics
        self.metric_auc = BinaryAUROC(thresholds=None)
        self.metric_auprc = BinaryAveragePrecision(thresholds=None)

        # Metrics tracker for plotting
        self.metrics_tracker = MetricsTracker()

        # OOF predictions storage
        self.oof_predictions = {"predictions": {}, "labels": {}}

        # Loss function - will be configured per ion with pos_weight
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

        # Gradient accumulation
        self.accum_steps = getattr(conf.training, "accum_steps", 1)

        if self.is_main_process:
            self.logger.info(f"E1BindingTrainer initialized:")
            self.logger.info(f"  - Device: {device}")
            self.logger.info(f"  - Use BF16: {self.use_bf16}")
            self.logger.info(f"  - Gradient accumulation steps: {self.accum_steps}")
            if is_distributed:
                self.logger.info(
                    f"  - Distributed: Yes (world_size={world_size}, rank={rank})"
                )

    def create_dataloaders(
        self,
        train_dataset,
        val_dataset,
        collate_fn,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
        """
        Create training and validation dataloaders.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            collate_fn: Collate function (E1DataCollatorForResidueClassification)
            batch_size: Batch size (defaults to config)
            num_workers: Number of data loading workers

        Returns:
            Tuple of (train_loader, val_loader, train_sampler)
            train_sampler is returned for setting epoch in distributed training
        """
        batch_size = batch_size or self.conf.training.batch_size

        # Create samplers for distributed training
        train_sampler = None
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            # Validation can be done on all processes (each sees different data)
            # or only on main process (simpler). We use all processes for speed.
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        else:
            val_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),  # Don't shuffle if using sampler
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader, train_sampler

    def _move_batch_to_device(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        ion: str,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            ion: Ion type being trained
            pos_weight: Positive class weight for BCE loss

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_loss_bce = 0.0
        total_loss_mlm = 0.0
        all_probs = []
        all_labels = []
        num_batches = 0

        # Move pos_weight to device if provided
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)

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
                    labels=batch["binding_labels"],
                    label_mask=batch["label_mask"],
                    pos_weight=pos_weight,
                    mlm_labels=batch["mlm_labels"],
                )

                loss = outputs.loss / self.accum_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += outputs.loss.item()
            if getattr(outputs, "loss_bce", None) is not None:
                total_loss_bce += float(outputs.loss_bce.detach())
            if getattr(outputs, "loss_mlm", None) is not None:
                total_loss_mlm += float(outputs.loss_mlm.detach())
            num_batches += 1

            # Collect predictions for metrics
            with torch.no_grad():
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                mask = batch["label_mask"]

                # Get valid predictions and labels
                valid_probs = probs[mask].detach().cpu()
                valid_labels = batch["binding_labels"][mask].detach().cpu()

                all_probs.append(valid_probs)
                all_labels.append(valid_labels)

        # Handle remaining gradients
        if num_batches % self.accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Compute metrics (convert to float32 for torchmetrics)
        all_probs = torch.cat(all_probs).float()
        all_labels = torch.cat(all_labels).long()

        train_metrics = {
            "loss": total_loss / num_batches,
            "loss_bce": total_loss_bce / max(num_batches, 1),
            "loss_mlm": total_loss_mlm / max(num_batches, 1),
            "auc": self.metric_auc(all_probs, all_labels).item(),
            "auprc": self.metric_auprc(all_probs, all_labels).item(),
        }

        # Track metrics
        self.metrics_tracker.update_train(train_metrics, ion)

        return train_metrics

    @torch.no_grad()
    def validate_epoch(
        self,
        val_loader: DataLoader,
        ion: str,
        fold_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader
            ion: Ion type being validated
            fold_idx: Fold index for OOF predictions storage

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
                    labels=batch["binding_labels"],
                    label_mask=batch["label_mask"],
                    mlm_labels=batch["mlm_labels"],
                )

                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                    if getattr(outputs, "loss_bce", None) is not None:
                        total_loss_bce += float(outputs.loss_bce)
                    if getattr(outputs, "loss_mlm", None) is not None:
                        total_loss_mlm += float(outputs.loss_mlm)
                    num_batches += 1

                # Collect predictions
                logits = outputs.logits
                probs = torch.sigmoid(logits)
                mask = batch["label_mask"]

                valid_probs = probs[mask].cpu()
                valid_labels = batch["binding_labels"][mask].cpu()

                all_probs.append(valid_probs)
                all_labels.append(valid_labels)

        # Compute metrics (convert to float32 for torchmetrics)
        all_probs = torch.cat(all_probs).float()
        all_labels = torch.cat(all_labels).long()

        val_auc = self.metric_auc(all_probs, all_labels).item()
        val_auprc = self.metric_auprc(all_probs, all_labels).item()

        # Find optimal threshold
        threshold_method = getattr(self.conf.training, "threshold_method", "youden")
        threshold_results = find_optimal_threshold(
            all_probs, all_labels, method=threshold_method
        )

        # Store OOF predictions
        if fold_idx is not None:
            self.store_oof_predictions(
                fold_idx, all_probs.numpy(), all_labels.numpy(), ion
            )

        val_metrics = {
            "loss": total_loss / max(num_batches, 1),
            "loss_bce": total_loss_bce / max(num_batches, 1),
            "loss_mlm": total_loss_mlm / max(num_batches, 1),
            "auc": val_auc,
            "auprc": val_auprc,
            "f1": threshold_results["f1"],
            "mcc": threshold_results["mcc"],
            "threshold": threshold_results["threshold"],
            "recall": threshold_results["recall"],
            "precision": threshold_results["precision"],
            "confusion_matrix": threshold_results["confusion_matrix"],
        }

        # Track metrics
        self.metrics_tracker.update_val(val_metrics, ion)

        return val_metrics

    def store_oof_predictions(
        self,
        fold_idx: int,
        predictions: np.ndarray,
        labels: np.ndarray,
        identifier: str,
    ):
        """Store out-of-fold predictions."""
        if fold_idx not in self.oof_predictions["predictions"]:
            self.oof_predictions["predictions"][fold_idx] = {}
            self.oof_predictions["labels"][fold_idx] = {}

        self.oof_predictions["predictions"][fold_idx][identifier] = predictions
        self.oof_predictions["labels"][fold_idx][identifier] = labels

    def setup_optimizer(
        self,
        model_params=None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
    ) -> torch.optim.Optimizer:
        """
        Setup optimizer.

        Args:
            model_params: Model parameters (defaults to self.model.parameters())
            lr: Learning rate (defaults to config)
            weight_decay: Weight decay (defaults to config)

        Returns:
            Configured optimizer
        """
        if model_params is None:
            model_params = self.model.parameters()

        lr = lr or self.conf.training.learning_rate
        weight_decay = weight_decay or getattr(self.conf.training, "weight_decay", 0.01)

        optimizer = torch.optim.AdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.logger.info(f"Optimizer: AdamW, lr={lr}, weight_decay={weight_decay}")

        return optimizer

    def setup_early_stopper(
        self,
        mode: str = "max",
        patience: Optional[int] = None,
        warmup_epochs: Optional[int] = None,
    ) -> EarlyStopper:
        """Setup early stopping."""
        patience = patience or getattr(self.conf.training, "early_stop_patience", 7)
        warmup_epochs = warmup_epochs or getattr(self.conf.training, "warmup_epochs", 3)

        return EarlyStopper(patience=patience, warmup_epochs=warmup_epochs, mode=mode)

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict] = None,
        additional_state: Optional[Dict] = None,
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optimizer to save state
            metrics: Metrics to include
            additional_state: Additional state to save
        """
        # Get the unwrapped model state dict (handles DDP wrapping)
        model_to_save = self._unwrapped_model

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        if additional_state is not None:
            checkpoint.update(additional_state)

        # Save classifier heads separately for easier loading
        if hasattr(model_to_save, "classifier_heads"):
            checkpoint["classifier_heads"] = model_to_save.classifier_heads.state_dict()

        torch.save(checkpoint, path)
        if self.is_main_process:
            self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info(f"Loaded checkpoint from {path}")
        return checkpoint

    def reset_metrics_tracker(self):
        """Reset metrics tracker for new fold."""
        self.metrics_tracker = MetricsTracker()

    def plot_training_curves(
        self,
        fold: Optional[int] = None,
        save_dir: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        """Plot and save training curves."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plots")
            return

        if save_dir is None:
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fold_str = str(fold) if fold is not None else "global"

        for ion in self.metrics_tracker.get_all_ions():
            ion_metrics = self.metrics_tracker.get_ion_metrics(ion)
            train_metrics = ion_metrics["train_metrics"]
            val_metrics = ion_metrics["val_metrics"]

            if not train_metrics or not val_metrics:
                continue

            # Plot loss
            plt.figure(figsize=(10, 6))
            plt.plot(
                [m["loss"] for m in train_metrics],
                label="Train Loss",
                marker="o",
                markersize=3,
            )
            plt.plot(
                [m["loss"] for m in val_metrics],
                label="Val Loss",
                marker="s",
                markersize=3,
            )
            plt.title(f"Loss - Fold {fold_str} [{ion}]")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_dir / f"loss_fold_{fold_str}_{ion}.png", dpi=150)
            plt.close()

            # Plot AUPRC
            plt.figure(figsize=(10, 6))
            plt.plot(
                [m["auprc"] for m in train_metrics],
                label="Train AUPRC",
                marker="o",
                markersize=3,
            )
            plt.plot(
                [m["auprc"] for m in val_metrics],
                label="Val AUPRC",
                marker="s",
                markersize=3,
            )
            plt.title(f"AUPRC - Fold {fold_str} [{ion}]")
            plt.xlabel("Epoch")
            plt.ylabel("AUPRC")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(save_dir / f"auprc_fold_{fold_str}_{ion}.png", dpi=150)
            plt.close()

        self.logger.info(f"Training curves saved to {save_dir}")
