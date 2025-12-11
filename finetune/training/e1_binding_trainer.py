"""
Trainer for E1 residue-level metal ion binding classification.

This module provides E1BindingTrainer which handles:
- Training with BCE loss and class balancing
- Validation with metrics computation (AUPRC, F1, MCC)
- OOF predictions storage for global threshold optimization
- Distributed Data Parallel (DDP) support for multi-GPU training
"""

import copy
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from .metrics import MetricsTracker, find_optimal_threshold, high_recall_auprc
from .visualization import plot_training_curves as viz_plot_training_curves

logger = logging.getLogger(__name__)


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
        pos_weights: Optional[Dict[str, torch.Tensor]] = None,
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
            pos_weights: Dictionary mapping ion type to positive class weight tensor
        """
        self.model = model
        self.conf = conf
        self.device = device
        self.logger = logger_instance or logger
        self.is_distributed = is_distributed
        self.world_size = world_size
        self.rank = rank
        self.is_main_process = rank == 0
        self.pos_weights = pos_weights or {}

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
        self.current_val_oof: Dict[str, Dict[str, Any]] = {}
        self.best_oof_predictions: Dict[str, Dict[str, Any]] = {}

        # Gradient accumulation
        self.accum_steps = getattr(conf.training, "accum_steps", 1)

        if self.is_main_process:
            self.logger.info("E1BindingTrainer initialized:")
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
        collate_fn=None,
        train_collate_fn=None,
        val_collate_fn=None,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
        """
        Create training and validation dataloaders.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            collate_fn: Collate function for both loaders (deprecated, use train_collate_fn/val_collate_fn)
            train_collate_fn: Collate function for training loader
            val_collate_fn: Collate function for validation loader
            batch_size: Batch size (defaults to config)
            num_workers: Number of data loading workers

        Returns:
            Tuple of (train_loader, val_loader, train_sampler)
            train_sampler is returned for setting epoch in distributed training
        """
        batch_size = batch_size or self.conf.training.batch_size

        # Handle backward compatibility: if single collate_fn provided, use for both
        if train_collate_fn is None:
            train_collate_fn = collate_fn
        if val_collate_fn is None:
            val_collate_fn = collate_fn

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
            collate_fn=train_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=val_collate_fn,
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

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)

            # Get pos_weight for this ion
            pos_weight = self.pos_weights.get(ion)
            if pos_weight is not None:
                pos_weight = pos_weight.to(self.device)

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
                    pos_weight=pos_weight,
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
        # Round floating point labels (from smoothing) back to integers for metrics
        all_labels = (torch.cat(all_labels) > 0.5).long()

        train_metrics = {
            "loss": total_loss / num_batches,
            "loss_bce": total_loss_bce / max(num_batches, 1),
            "loss_mlm": total_loss_mlm / max(num_batches, 1),
            "auc": self.metric_auc(all_probs, all_labels).item(),
            "auprc": self.metric_auprc(all_probs, all_labels).item(),
            "high_recall_auprc": high_recall_auprc(
                all_labels, all_probs, recall_threshold=0.7
            ),
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

        # For OOF capture on the main process only
        collecting_oof = self.is_main_process and fold_idx is not None

        for batch in val_loader:
            batch = self._move_batch_to_device(batch)

            # Get pos_weight for this ion
            pos_weight = self.pos_weights.get(ion)
            if pos_weight is not None:
                pos_weight = pos_weight.to(self.device)

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
                    mlm_labels=batch.get("mlm_labels"),
                    pos_weight=pos_weight,
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

                valid_probs = probs[mask].detach().cpu()
                valid_labels = batch["binding_labels"][mask].detach().cpu()

                all_probs.append(valid_probs)
                all_labels.append(valid_labels)

                if collecting_oof:
                    self._accumulate_oof_from_batch(
                        ion=ion,
                        probs=probs,
                        labels=batch["binding_labels"],
                        label_mask=mask,
                        protein_ids=batch.get("protein_ids"),
                        fold_idx=fold_idx,
                    )

        # Compute metrics (convert to float32 for torchmetrics)
        all_probs = torch.cat(all_probs).float()
        # Round floating point labels (from smoothing) back to integers for metrics
        all_labels = torch.cat(all_labels).long()

        val_auc = self.metric_auc(all_probs, all_labels).item()
        val_auprc = self.metric_auprc(all_probs, all_labels).item()
        val_hra = high_recall_auprc(all_labels, all_probs, recall_threshold=0.7)

        # Find optimal threshold
        threshold_method = getattr(self.conf.training, "threshold_method", "youden")
        threshold_results = find_optimal_threshold(
            all_probs, all_labels, method=threshold_method
        )

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

        # Track metrics
        self.metrics_tracker.update_val(val_metrics, ion)

        return val_metrics

    def _accumulate_oof_from_batch(
        self,
        ion: str,
        probs: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        protein_ids: Optional[Any],
        fold_idx: int,
    ):
        """Collect per-residue OOF records for the current validation epoch."""
        # Ensure we have identifiers for each sample in batch
        batch_size = probs.shape[0]
        if protein_ids is None or len(protein_ids) != batch_size:
            protein_ids = [f"sample_{i}" for i in range(batch_size)]

        entry = self.current_val_oof.setdefault(
            ion, {"ids": [], "labels": [], "probs": [], "fold": fold_idx}
        )

        for sample_idx, protein_id in enumerate(protein_ids):
            sample_mask = label_mask[sample_idx].detach().cpu()
            if not torch.any(sample_mask):
                continue

            sample_probs = (
                probs[sample_idx][sample_mask]
                .detach()
                .float()  # ensure numpy-friendly dtype
                .cpu()
                .numpy()
                .ravel()
                .tolist()
            )
            sample_labels = (
                labels[sample_idx][sample_mask]
                .detach()
                .float()
                .cpu()
                .numpy()
                .ravel()
                .tolist()
            )

            # Positions are in the same order as the valid residues
            residue_ids = [f"{protein_id}:{pos}" for pos in range(len(sample_labels))]

            entry["ids"].extend(residue_ids)
            entry["labels"].extend(sample_labels)
            entry["probs"].extend(sample_probs)
            entry["fold"] = fold_idx

    def reset_oof_tracking(self):
        """Reset all OOF tracking structures (used per fold)."""
        self.current_val_oof = {}
        self.best_oof_predictions = {}

    def reset_current_oof(self):
        """Reset OOF buffers for the current epoch."""
        self.current_val_oof = {}

    def snapshot_current_oof_as_best(self):
        """Persist the current epoch's OOF buffers as the fold-best snapshot."""
        self.best_oof_predictions = {}
        for ion, data in self.current_val_oof.items():
            self.best_oof_predictions[ion] = {
                "ids": np.array(data.get("ids", []), dtype=object),
                "labels": np.array(data.get("labels", []), dtype=float),
                "probs": np.array(data.get("probs", []), dtype=float),
                "fold": data.get("fold"),
            }

    def get_best_oof(self) -> Dict[str, Dict[str, Any]]:
        """Return the best OOF snapshot (per ion) for the fold."""
        return copy.deepcopy(self.best_oof_predictions)

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

        # Prefer fused AdamW on CUDA for better throughput; fall back gracefully
        fused_kwargs = {"fused": True} if torch.cuda.is_available() else {}
        fused_used = False
        try:
            optimizer = torch.optim.AdamW(
                model_params,
                lr=lr,
                weight_decay=weight_decay,
                **fused_kwargs,
            )
            fused_used = fused_kwargs.get("fused", False)
        except TypeError:
            if fused_kwargs:
                self.logger.warning(
                    "torch.optim.AdamW does not support fused=True in this environment; "
                    "falling back to standard AdamW"
                )
            optimizer = torch.optim.AdamW(
                model_params,
                lr=lr,
                weight_decay=weight_decay,
            )

        fused_msg = " (fused)" if fused_used else ""
        self.logger.info(
            f"Optimizer: AdamW{fused_msg}, lr={lr}, weight_decay={weight_decay}"
        )

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
        Save LoRA-only checkpoint (adapters + classifier heads, not full base model).

        This saves ~99% less disk space than saving the full model state:
        - LoRA adapters: ~2-3MB (saved via PEFT's save_pretrained)
        - Classifier heads: ~100KB
        - Total: ~3-5MB instead of ~1.2GB per fold

        Directory structure:
            {path}_lora/         # PEFT adapter directory
                adapter_config.json
                adapter_model.safetensors
            {path}_heads.pt      # Classifier heads + metadata

        Args:
            path: Base path for checkpoint (without extension)
            epoch: Current epoch
            optimizer: Optimizer to save state
            metrics: Metrics to include
            additional_state: Additional state to save
        """
        model = self._unwrapped_model

        # Remove .pt extension if present for clean directory naming
        base_path = str(path).replace(".pt", "")
        lora_dir = f"{base_path}_lora"
        heads_path = f"{base_path}_heads.pt"

        # Save LoRA adapters using PEFT's save_pretrained
        # This saves only the adapter weights + config, not the base model
        if hasattr(model, "_original_model"):
            peft_model = model._original_model
            peft_model.save_pretrained(lora_dir)
            if self.is_main_process:
                self.logger.info(f"LoRA adapters saved to {lora_dir}")
        else:
            self.logger.warning(
                "Model does not have _original_model attribute. "
                "Falling back to full state dict save."
            )
            torch.save({"model_state_dict": model.state_dict()}, f"{base_path}_full.pt")
            return

        # Save classifier heads + metadata
        checkpoint = {
            "epoch": epoch,
            "base_model_checkpoint": getattr(self.conf.model, "checkpoint", None),
            "lora_config": dict(self.conf.lora) if hasattr(self.conf, "lora") else None,
        }

        if hasattr(model, "classifier_heads"):
            checkpoint["classifier_heads"] = model.classifier_heads.state_dict()
            checkpoint["ions"] = list(model.classifier_heads.keys())

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        if additional_state is not None:
            checkpoint.update(additional_state)

        torch.save(checkpoint, heads_path)
        if self.is_main_process:
            self.logger.info(f"Classifier heads saved to {heads_path}")

    def load_checkpoint(self, path: str) -> Dict:
        """
        Load LoRA-only checkpoint.

        Note: This only loads the heads checkpoint metadata.
        For full model reconstruction, use load_lora_checkpoint() from e1_checkpoint_utils.

        Args:
            path: Base path to checkpoint (without _heads.pt suffix)

        Returns:
            Checkpoint dictionary with epoch, metrics, etc.
        """
        base_path = str(path).replace(".pt", "").replace("_heads", "")
        heads_path = f"{base_path}_heads.pt"

        checkpoint = torch.load(heads_path, map_location=self.device)

        # Load classifier heads if present
        if "classifier_heads" in checkpoint and hasattr(
            self._unwrapped_model, "classifier_heads"
        ):
            self._unwrapped_model.classifier_heads.load_state_dict(
                checkpoint["classifier_heads"]
            )

        self.logger.info(f"Loaded checkpoint from {heads_path}")
        return checkpoint

    def reset_metrics_tracker(self):
        """Reset metrics tracker for new fold."""
        self.metrics_tracker = MetricsTracker()
        self.reset_oof_tracking()

    def plot_training_curves(
        self,
        fold: Optional[int] = None,
        save_dir: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        """Plot and save training curves."""
        if save_dir is None:
            return
        viz_plot_training_curves(
            metrics_tracker=self.metrics_tracker,
            fold=fold,
            save_dir=save_dir,
            stage=stage,
        )
