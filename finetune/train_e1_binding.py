"""
E1 LoRA-based finetuning for residue-level metal ion binding classification.

This script provides direct task-aligned finetuning of the E1 model with:
- LoRA adapters for efficient finetuning
- MSA context from homolog sequences
- Ion-specific classification heads
- 5-fold cross-validation
- Distributed Data Parallel (DDP) support for multi-GPU training

Usage:
    Single GPU:
        python train_e1_binding.py --config finetune/configs/e1_binding_config.yaml

    Multi-GPU (DDP via torchrun):
        torchrun --nproc_per_node=4 train_e1_binding.py --config finetune/configs/e1_binding_config.yaml
"""

import os
import sys

# Add src directory to Python path so E1 module can be imported without installing the package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import numpy as np
from pathlib import Path
from timeit import default_timer as timer
from training.finetune_utils import (
    set_seeds,
    setup_logging,
    process_config,
)
from training.e1_classification_model import E1ForResidueClassification
from training.e1_binding_dataset import create_binding_datasets_from_config
from training.e1_binding_collator import E1DataCollatorForResidueClassification
from training.e1_joint_collator import E1DataCollatorForJointBindingMLM
from training.e1_binding_trainer import E1BindingTrainer
from training.e1_finetune_utils import load_e1_model
from training.e1_joint_model import E1ForJointBindingMLM

logger = logging.getLogger(__name__)


def is_distributed():
    """Check if running in distributed mode."""
    return os.environ.get("RANK") is not None


def get_rank():
    """Get current process rank (0 if not distributed)."""
    if is_distributed():
        return int(os.environ.get("RANK", 0))
    return 0


def get_local_rank():
    """Get local rank for device assignment."""
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def get_world_size():
    """Get total number of processes."""
    if is_distributed():
        return int(os.environ.get("WORLD_SIZE", 1))
    return 1


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed():
    """Initialize distributed training environment."""
    if not is_distributed():
        return None

    # Initialize process group
    dist.init_process_group(backend="nccl")

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        # Ensure min_lr is a float (handle case where it comes from config as string)
        self.min_lr = float(min_lr)
        # Ensure base_lrs are floats (handle case where lr comes from config as string)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self, metrics=None):
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            factor = self.current_epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            # Ensure both values are floats before comparison
            new_lr = max(float(base_lr) * float(factor), float(self.min_lr))
            param_group["lr"] = new_lr


def load_e1_for_classification(
    checkpoint: str,
    lora_config: dict,
    ion_types: list,
    dropout: float = 0.1,
    model_dtype: str = None,
    device: torch.device = None,
    mlm_weight: float = 0.0,
) -> E1ForResidueClassification:
    """
    Load E1 model with LoRA and wrap with classification heads.

    Args:
        checkpoint: E1 checkpoint path
        lora_config: LoRA configuration dictionary
        ion_types: List of ion types for classification heads
        dropout: Dropout rate for classification heads
        model_dtype: Model dtype ("bfloat16", "float16", or None)
        device: Device to move model to

    Returns:
        E1ForResidueClassification model with LoRA adapters
    """
    # Load E1 model with LoRA
    e1_model, batch_preparer = load_e1_model(
        checkpoint=checkpoint,
        lora_config=lora_config,
        model_dtype=model_dtype,
    )

    # Wrap with classification heads
    if mlm_weight and mlm_weight > 0:
        model = E1ForJointBindingMLM(
            e1_model=e1_model,
            ion_types=ion_types,
            dropout=dropout,
            mlm_weight=mlm_weight,
            freeze_backbone=False,  # LoRA handles selective training
        )
    else:
        model = E1ForResidueClassification(
            e1_model=e1_model,
            ion_types=ion_types,
            dropout=dropout,
            freeze_backbone=False,  # LoRA handles selective training
        )

    # Move to device and dtype
    if device is not None:
        dtype = None
        if model_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif model_dtype == "float16":
            dtype = torch.float16
        model = model.to(device=device, dtype=dtype)

    return model, batch_preparer


def train_single_fold(conf, fold_idx: int, base_output_path: str):
    """
    Train a single fold of K-fold cross-validation.

    Args:
        conf: Configuration object
        fold_idx: Current fold index (1-based)
        base_output_path: Base output path for saving models

    Returns:
        dict: Results for this fold
    """
    seed = int(conf.general.seed)
    set_seeds(seed)

    # Setup device based on distributed mode
    if is_distributed():
        local_rank = get_local_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(
            f"cuda:{conf.general.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

    if is_main_process():
        logging.info(f"\n{'='*100}")
        logging.info(
            f"Fold {fold_idx} Training begins at {datetime.datetime.now().strftime('%m-%d %H:%M')}"
        )
        logging.info(f"{'='*100}")

    # Get ion list
    ion_list = conf.training.ions

    # Load model
    model, batch_preparer = load_e1_for_classification(
        checkpoint=conf.model.checkpoint,
        lora_config=dict(conf.lora),
        ion_types=ion_list,
        dropout=getattr(conf.model, "dropout", 0.1),
        model_dtype=getattr(conf.training, "model_dtype", "bfloat16"),
        device=device,
        mlm_weight=getattr(conf.training, "mlm_weight", 0.0),
    )

    # Wrap model with DDP if distributed
    if is_distributed():
        # Find unused parameters is needed when not all parameters are used in every forward pass
        # This is common with multi-head models where only one head is used at a time
        ddp_find_unused = getattr(conf.training, "ddp_find_unused_parameters", True)
        model = DDP(
            model, device_ids=[get_local_rank()], find_unused_parameters=ddp_find_unused
        )
        if is_main_process():
            logging.info(
                f"Model wrapped with DDP (find_unused_parameters={ddp_find_unused})"
            )

    if is_main_process():
        logging.info(f"Model loaded: E1ForResidueClassification with ions {ion_list}")
        logging.info(f"  - Device: {device}")
        if is_distributed():
            logging.info(f"  - World size: {get_world_size()}")

    # Create data collator
    msa_config = conf.training.get("msa_sampling", {})
    mlm_weight = getattr(conf.training, "mlm_weight", 0.0)
    if mlm_weight and mlm_weight > 0:
        data_collator = E1DataCollatorForJointBindingMLM(
            mlm_probability=getattr(conf.training, "mlm_probability", 0.15),
            max_total_tokens=msa_config.get("max_token_length", 8192),
            max_query_tokens=msa_config.get("max_query_length", 1024),
            ignore_index=-100,
        )
    else:
        data_collator = E1DataCollatorForResidueClassification(
            max_total_tokens=msa_config.get("max_token_length", 8192),
            max_query_tokens=msa_config.get("max_query_length", 1024),
        )

    # Prepare datasets for each ion
    if is_main_process():
        logging.info(
            f"Preparing datasets for ions: {ion_list} (Fold {fold_idx} as validation)..."
        )
    datasets = {}
    pos_weights = {}

    for ion in ion_list:
        train_dataset, val_dataset, pos_weight = create_binding_datasets_from_config(
            data_conf=dict(conf.data),
            training_conf=dict(conf.training),
            ion_type=ion,
            fold_to_use_as_val=fold_idx,
            seed=seed,
        )
        datasets[ion] = (train_dataset, val_dataset)
        pos_weights[ion] = pos_weight

        if is_main_process():
            logging.info(
                f"  {ion}: train={len(train_dataset)}, val={len(val_dataset)}, pos_weight={pos_weight.item():.2f}"
            )

    # Initialize trainer (pass DDP info)
    trainer = E1BindingTrainer(
        model,
        conf,
        device,
        logging,
        is_distributed=is_distributed(),
        world_size=get_world_size(),
        rank=get_rank(),
    )

    # Create dataloaders with DistributedSampler if needed
    train_loaders = {}
    val_loaders = {}
    train_samplers = {}  # Keep track of samplers for epoch setting
    batch_size = conf.training.batch_size
    num_workers = getattr(conf.training, "num_workers", 4)

    for ion in ion_list:
        train_dataset, val_dataset = datasets[ion]
        train_loader, val_loader, train_sampler = trainer.create_dataloaders(
            train_dataset,
            val_dataset,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        train_loaders[ion] = train_loader
        val_loaders[ion] = val_loader
        train_samplers[ion] = train_sampler

    if is_main_process():
        logging.info(f"\n{'='*80}")
        logging.info("Starting E1 Binding Classification Training")
        logging.info(f"{'='*80}")

    # Training configuration
    num_epochs = conf.training.epochs

    # Setup optimizer
    optimizer = trainer.setup_optimizer()

    # Learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=getattr(conf.training, "warmup_epochs", 3),
        total_epochs=num_epochs,
        min_lr=getattr(conf.training, "min_lr", 1e-6),
    )

    # Early stopping
    early_stopper = trainer.setup_early_stopper(mode="max")

    # Reset metrics tracker
    trainer.reset_metrics_tracker()

    # Training loop
    best_auprc_per_ion = {ion: 0.0 for ion in ion_list}
    best_thresholds = {ion: {"f1": 0.5, "mcc": 0.5} for ion in ion_list}
    best_total_auprc = 0.0
    best_model_path = None

    for epoch in range(1, num_epochs + 1):
        total_train_auprc = 0.0
        total_val_auprc = 0.0

        # Update dataset epoch for MSA sampling variation
        for ion in ion_list:
            datasets[ion][0].set_epoch(epoch)
            # Set epoch for distributed sampler if present
            if train_samplers[ion] is not None:
                train_samplers[ion].set_epoch(epoch)

        for ion in ion_list:
            # Train
            train_metrics = trainer.train_epoch(
                train_loaders[ion],
                optimizer,
                ion,
                pos_weight=pos_weights[ion],
            )

            # Validate (only on main process to avoid redundant computation)
            val_metrics = trainer.validate_epoch(
                val_loaders[ion],
                ion,
                fold_idx=fold_idx,
            )

            if is_main_process():
                train_loss_bce = train_metrics.get("loss_bce")
                train_loss_mlm = train_metrics.get("loss_mlm")
                val_loss_bce = val_metrics.get("loss_bce")
                val_loss_mlm = val_metrics.get("loss_mlm")

                train_loss_bce_str = (
                    f"{train_loss_bce:.4f}" if train_loss_bce is not None else "n/a"
                )
                train_loss_mlm_str = (
                    f"{train_loss_mlm:.4f}" if train_loss_mlm is not None else "n/a"
                )
                val_loss_bce_str = (
                    f"{val_loss_bce:.4f}" if val_loss_bce is not None else "n/a"
                )
                val_loss_mlm_str = (
                    f"{val_loss_mlm:.4f}" if val_loss_mlm is not None else "n/a"
                )

                logging.info(
                    f"Epoch {epoch}/{num_epochs} [{ion}] - "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"(BCE: {train_loss_bce_str}, MLM: {train_loss_mlm_str}), "
                    f"AUC: {train_metrics['auc']:.3f}, AUPRC: {train_metrics['auprc']:.3f}"
                )

                cm = val_metrics["confusion_matrix"]
                logging.info(
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"(BCE: {val_loss_bce_str}, MLM: {val_loss_mlm_str}), "
                    f"AUC: {val_metrics['auc']:.3f}, AUPRC: {val_metrics['auprc']:.3f}, "
                    f"MCC: {val_metrics['mcc']:.3f} @ {val_metrics['threshold']:.3f}, "
                    f"F1: {val_metrics['f1']:.3f} @ {val_metrics['threshold']:.3f}"
                )
                logging.info(
                    f"Recall: {val_metrics['recall']:.3f}, Precision: {val_metrics['precision']:.3f}"
                )
                logging.info(
                    f"TN: {cm[0][0]}, FP: {cm[0][1]}, FN: {cm[1][0]}, TP: {cm[1][1]}\n"
                )

            total_train_auprc += train_metrics["auprc"]
            total_val_auprc += val_metrics["auprc"]

            if val_metrics["auprc"] > best_auprc_per_ion[ion]:
                best_auprc_per_ion[ion] = val_metrics["auprc"]
                best_thresholds[ion]["f1"] = val_metrics["threshold"]
                best_thresholds[ion]["mcc"] = val_metrics["threshold"]

        # Average validation AUPRC across all ions
        avg_val_auprc = total_val_auprc / len(ion_list)

        # Save best model (only on main process)
        save_model = getattr(conf.training, "save_model", True)
        if avg_val_auprc > best_total_auprc:
            best_total_auprc = avg_val_auprc
            if save_model and is_main_process():
                fold_output_path = f"{base_output_path}/fold_{fold_idx}"
                Path(fold_output_path).mkdir(parents=True, exist_ok=True)
                best_model_path = f"{fold_output_path}/best_e1_binding_model.pt"

                trainer.save_checkpoint(
                    path=best_model_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    metrics={"auprc": avg_val_auprc},
                    additional_state={
                        "ions": ion_list,
                        "best_thresholds": best_thresholds,
                        "fold": fold_idx,
                    },
                )
                logging.info(
                    f"New best model at epoch {epoch} with AUPRC: {best_total_auprc:.3f} (saved)"
                )
            elif is_main_process():
                logging.info(
                    f"New best model at epoch {epoch} with AUPRC: {best_total_auprc:.3f} (not saved)"
                )

        # Learning rate scheduling
        scheduler.step()

        # Early stopping
        if early_stopper.early_stop(avg_val_auprc):
            if is_main_process():
                logging.info(f"Early stopping at epoch {epoch}")
            break

        # Synchronize across processes (barrier) before next epoch
        if is_distributed():
            dist.barrier()

    # Generate training curves (only on main process)
    if is_main_process():
        fold_plots_dir = f"{base_output_path}/fold_{fold_idx}/plots"
        trainer.plot_training_curves(fold=fold_idx, save_dir=fold_plots_dir)
        logging.info(f"Training curves saved to {fold_plots_dir}")

    # Return results
    fold_results = {
        "fold": fold_idx,
        "best_auprc_per_ion": best_auprc_per_ion,
        "best_total_auprc": best_total_auprc,
        "best_thresholds": best_thresholds,
        "model_path": best_model_path,
        "oof_predictions": trainer.oof_predictions.copy(),
    }

    if is_main_process():
        logging.info(f"\nFold {fold_idx} Training Summary")
        logging.info(f"{'='*50}")
        for ion in ion_list:
            logging.info(f"{ion}: Best AUPRC = {best_auprc_per_ion[ion]:.3f}")
        logging.info(f"Average AUPRC: {best_total_auprc:.3f}")
        if best_model_path is not None:
            logging.info(f"Best model saved to: {best_model_path}")

    return fold_results


def run_cv(conf, train_fn, base_output_path: str):
    """
    Run K-fold cross-validation.

    Args:
        conf: Configuration object
        train_fn: Training function
        base_output_path: Base output path

    Returns:
        List of fold results
    """
    if is_main_process():
        logging.info(
            f"K-fold cross-validation training begins at {datetime.datetime.now().strftime('%m-%d %H:%M')}"
        )

    num_folds = getattr(conf.training, "num_folds", 5)
    if is_main_process():
        logging.info(f"Running {num_folds}-fold cross-validation")
        if is_distributed():
            logging.info(f"  - Distributed training with {get_world_size()} GPUs")

    Path(base_output_path).mkdir(parents=True, exist_ok=True)

    all_fold_results = []

    for fold_idx in range(1, num_folds + 1):
        if is_main_process():
            logging.info(f"\n{'='*120}")
            logging.info(f"Starting Fold {fold_idx}/{num_folds}")
            logging.info(f"{'='*120}")

        fold_results = train_fn(conf, fold_idx, base_output_path)
        all_fold_results.append(fold_results)

        # Synchronize across processes before next fold
        if is_distributed():
            dist.barrier()

    # Log summary (only on main process)
    if is_main_process():
        log_cv_summary(all_fold_results, conf)

        logging.info(
            f"\nK-fold cross-validation completed at {datetime.datetime.now().strftime('%m-%d %H:%M')}"
        )

    return all_fold_results


def log_cv_summary(all_fold_results: list, conf):
    """Log cross-validation summary."""
    logging.info(f"\n{'='*120}")
    logging.info("K-FOLD CROSS-VALIDATION SUMMARY")
    logging.info(f"{'='*120}")

    ion_list = conf.training.ions
    fold_auprcs = {ion: [] for ion in ion_list}

    for fold_result in all_fold_results:
        logging.info(f"\nFold {fold_result['fold']}:")
        for ion in ion_list:
            auprc = fold_result["best_auprc_per_ion"][ion]
            fold_auprcs[ion].append(auprc)
            logging.info(f"  {ion}: AUPRC = {auprc:.3f}")
        logging.info(f"  Average AUPRC: {fold_result['best_total_auprc']:.3f}")
        if fold_result.get("model_path"):
            logging.info(f"  Model saved: {fold_result['model_path']}")

    # Calculate mean and std across folds
    logging.info(f"\n{'='*60}")
    logging.info("AVERAGE PERFORMANCE ACROSS FOLDS:")
    logging.info(f"{'='*60}")

    for ion in ion_list:
        mean_auprc = np.mean(fold_auprcs[ion])
        std_auprc = np.std(fold_auprcs[ion])
        logging.info(f"{ion}: {mean_auprc:.3f} ± {std_auprc:.3f}")

    overall_mean = np.mean([result["best_total_auprc"] for result in all_fold_results])
    overall_std = np.std([result["best_total_auprc"] for result in all_fold_results])
    logging.info(f"Average AUPRC: {overall_mean:.3f} ± {overall_std:.3f}")


def main(conf):
    """Main function implementing K-fold cross-validation."""
    base_output_path = str(conf.output_path)
    return run_cv(conf, train_single_fold, base_output_path)


if __name__ == "__main__":
    start = timer()

    parser = argparse.ArgumentParser(
        description="E1 LoRA finetuning for metal ion binding classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="finetune/configs/e1_binding_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    # Setup distributed training if launched with torchrun
    local_rank = setup_distributed()

    # Load config
    config_path = Path(args.config)
    config_name = config_path.stem

    conf, output_path = process_config(args.config, config_name=config_name)

    # Setup logging (only on main process to avoid duplicate logs)
    if is_main_process():
        setup_logging(str(output_path))

        logging.info("=" * 80)
        logging.info("E1 Binding Classification Fine-tuning")
        logging.info("=" * 80)
        logging.info(f"Config: {args.config}")
        logging.info(f"Output: {output_path}")
        if is_distributed():
            logging.info(f"Distributed training: {get_world_size()} GPUs")

    try:
        main(conf)
    finally:
        # Cleanup distributed training
        cleanup_distributed()

    end = timer()
    if is_main_process():
        logging.info(f"Total time: {(end - start) / 60:.1f} minutes")
