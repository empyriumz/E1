"""
E1 contrastive learning for residue-level metal ion binding classification.

Training pipeline with:
- Unsupervised Contrastive Learning (UCL) from masked variants
- Prototype alignment for class separation
- BCE classification using prototype distances
- Auxiliary MLM loss for regularization
- LoRA finetuning and K-fold cross-validation

Usage:
    Single GPU:
        python train_e1_contrastive.py --config finetune/configs/e1_contrastive.yaml

    Multi-GPU (DDP):
        torchrun --nproc_per_node=4 train_e1_contrastive.py --config finetune/configs/e1_contrastive.yaml
"""

import os
import sys

# Add src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import datetime
import logging
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
import torch.distributed as dist
from loss_func_contrastive import PrototypeBCELoss
from modeling_e1 import compile_flex_attention_if_enabled
from torch.nn.parallel import DistributedDataParallel as DDP
from training.e1_binding_dataset import (
    compute_shared_ree_splits,
    create_binding_datasets_from_config,
)
from training.e1_contrastive_collator import E1DataCollatorForContrastive
from training.e1_contrastive_model import E1ForContrastiveBinding
from training.e1_contrastive_trainer import E1ContrastiveTrainer
from training.e1_finetune_utils import load_e1_model
from training.finetune_utils import process_config, set_seeds, setup_logging
from training.distributed_utils import (
    is_distributed,
    get_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
)
from training.scheduler import WarmupCosineScheduler
from training.cv_utils import run_cv

logger = logging.getLogger(__name__)


# ============================================================================
# Model loading
# ============================================================================


def load_e1_for_contrastive(
    checkpoint: str,
    lora_config: dict,
    ion_types: list,
    contrastive_config: dict,
    dropout: float = 0.1,
    model_dtype: str = None,
    device: torch.device = None,
    mlm_weight: float = 0.4,
):
    """Load E1 model with LoRA and wrap with contrastive learning components."""
    # Load E1 model with LoRA
    e1_model, batch_preparer = load_e1_model(
        checkpoint=checkpoint, lora_config=lora_config, model_dtype=model_dtype
    )

    # Wrap with contrastive model
    model = E1ForContrastiveBinding(
        e1_model=e1_model,
        ion_types=ion_types,
        dropout=dropout,
        prototype_dim=contrastive_config.get("prototype_dim"),
        use_ema_prototypes=contrastive_config.get("use_ema_prototypes", True),
        ema_decay=contrastive_config.get("ema_decay", 0.999),
        mlm_weight=mlm_weight,
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


# ============================================================================
# Training loop
# ============================================================================


def train_single_fold(conf, fold_idx: int, base_output_path: str):
    """Train a single fold with contrastive learning."""
    seed = int(conf.general.seed)
    set_seeds(seed)

    # Setup device
    if is_distributed():
        local_rank = get_local_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(
            f"cuda:{conf.general.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

    if is_main_process():
        logging.info(f"\n{'=' * 100}")
        logging.info(
            f"Fold {fold_idx} Contrastive Training at {datetime.datetime.now().strftime('%m-%d %H:%M')}"
        )
        logging.info(f"{'=' * 100}")

    ion_list = conf.training.ions
    contrastive_conf = dict(conf.contrastive)

    # Compile flex_attention if enabled
    compile_flex_attention = getattr(conf.training, "compile_flex_attention", False)
    if compile_flex_attention:
        if is_main_process():
            logging.info("Enabling torch.compile for flex_attention")
        compile_flex_attention_if_enabled(enabled=True)

    # Load model
    model, batch_preparer = load_e1_for_contrastive(
        checkpoint=conf.model.checkpoint,
        lora_config=dict(conf.lora),
        ion_types=ion_list,
        contrastive_config=contrastive_conf,
        dropout=getattr(conf.model, "dropout", 0.1),
        model_dtype=getattr(conf.training, "model_dtype", "bfloat16"),
        device=device,
        mlm_weight=getattr(conf.training, "mlm_weight", 0.4),
    )

    # Wrap with DDP if distributed
    if is_distributed():
        ddp_find_unused = getattr(conf.training, "ddp_find_unused_parameters", True)
        model = DDP(
            model, device_ids=[get_local_rank()], find_unused_parameters=ddp_find_unused
        )
        if is_main_process():
            logging.info(
                f"Model wrapped with DDP (find_unused_parameters={ddp_find_unused})"
            )

    if is_main_process():
        logging.info(f"Model: E1ForContrastiveBinding with ions {ion_list}")
        logging.info(f"  - Device: {device}")

    # Create contrastive collator
    msa_config = conf.training.get("msa_sampling", {})
    val_msa_config = conf.training.get("validation_msa_sampling", msa_config)

    train_collator = E1DataCollatorForContrastive(
        num_variants=contrastive_conf.get("num_variants", 4),
        mask_prob_min=contrastive_conf.get("mask_prob_min", 0.05),
        mask_prob_max=contrastive_conf.get("mask_prob_max", 0.15),
        max_total_tokens=msa_config.get("max_token_length", 6144),
        max_query_tokens=msa_config.get("max_query_length", 768),
    )

    # Validation collator with single variant for consistent metrics
    val_collator = E1DataCollatorForContrastive(
        num_variants=1,  # Single view for validation (faster, consistent)
        mask_prob_min=0.0,  # No masking for validation
        mask_prob_max=0.0,
        max_total_tokens=val_msa_config.get("max_token_length", 8192),
        max_query_tokens=val_msa_config.get("max_query_length", 1024),
    )

    # Create loss function
    loss_fn = PrototypeBCELoss(
        temperature=contrastive_conf.get("temperature", 0.07),
        eps=contrastive_conf.get("eps", 0.1),
        eps_pos=contrastive_conf.get("eps_pos"),
        eps_neg=contrastive_conf.get("eps_neg"),
        prototype_weight=contrastive_conf.get("prototype_weight", 1.0),
        unsupervised_weight=contrastive_conf.get("unsupervised_weight", 0.5),
        bce_weight=contrastive_conf.get("bce_weight", 1.0),
        scoring_temperature=contrastive_conf.get("scoring_temperature", 1.0),
        label_smoothing=contrastive_conf.get("label_smoothing", 0.0),
        device=device,
        logger=logger,
    )

    # Prepare datasets
    if is_main_process():
        logging.info(
            f"Preparing datasets for ions: {ion_list} (Fold {fold_idx} as validation)"
        )

    datasets = {}

    # Handle REE-related ions
    ree_related_ions = {"LREE", "HREE", "REE"}
    has_ree_related = bool(ree_related_ions & set(ion_list))
    shared_train_ids = None
    shared_val_ids = None

    if has_ree_related:
        shared_train_ids, shared_val_ids = compute_shared_ree_splits(
            data_conf=dict(conf.data),
            training_conf=dict(conf.training),
            fold_to_use_as_val=fold_idx,
            seed=seed,
        )

    for ion in ion_list:
        if ion in ree_related_ions:
            train_dataset, val_dataset, class_counts = (
                create_binding_datasets_from_config(
                    data_conf=dict(conf.data),
                    training_conf=dict(conf.training),
                    ion_type=ion,
                    fold_to_use_as_val=fold_idx,
                    seed=seed,
                    train_ids=shared_train_ids,
                    val_ids=shared_val_ids,
                )
            )
        else:
            train_dataset, val_dataset, class_counts = (
                create_binding_datasets_from_config(
                    data_conf=dict(conf.data),
                    training_conf=dict(conf.training),
                    ion_type=ion,
                    fold_to_use_as_val=fold_idx,
                    seed=seed,
                )
            )

        datasets[ion] = (train_dataset, val_dataset)

        if is_main_process():
            pos_count, neg_count = class_counts
            logging.info(
                f"  {ion}: train={len(train_dataset)}, val={len(val_dataset)}, pos={pos_count}, neg={neg_count}"
            )

    # Initialize trainer
    trainer = E1ContrastiveTrainer(
        model=model,
        conf=conf,
        device=device,
        loss_fn=loss_fn,
        logger_instance=logging,
        is_distributed=is_distributed(),
        world_size=get_world_size(),
        rank=get_rank(),
    )

    # Create dataloaders
    train_loaders = {}
    val_loaders = {}
    train_samplers = {}
    batch_size = conf.training.batch_size
    num_workers = getattr(conf.training, "num_workers", 2)

    for ion in ion_list:
        train_dataset, val_dataset = datasets[ion]
        train_loader, val_loader, train_sampler = trainer.create_dataloaders(
            train_dataset,
            val_dataset,
            train_collate_fn=train_collator,
            val_collate_fn=val_collator,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        train_loaders[ion] = train_loader
        val_loaders[ion] = val_loader
        train_samplers[ion] = train_sampler

        # Initialize prototypes
        trainer.initialize_prototypes(train_loader, ion)

    if is_main_process():
        logging.info(f"\n{'=' * 80}")
        logging.info("Starting Contrastive Training")
        logging.info(f"{'=' * 80}")

    # Training configuration
    num_epochs = conf.training.epochs
    optimizer = trainer.setup_optimizer()

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=getattr(conf.training, "warmup_epochs", 3),
        total_epochs=num_epochs,
        min_lr=getattr(conf.training, "min_lr", 1e-6),
    )

    early_stopper = trainer.setup_early_stopper(mode="max")
    trainer.reset_metrics_tracker()

    # Training loop
    best_epoch_metrics = None
    best_total_hra = 0.0
    best_total_auprc = 0.0
    best_model_path = None
    val_metrics_per_ion = {}

    for epoch in range(1, num_epochs + 1):
        total_train_hra = 0.0
        total_val_hra = 0.0
        total_val_auprc = 0.0

        trainer.reset_current_oof()

        # Update MSA sampling for epoch
        for ion in ion_list:
            datasets[ion][0].set_epoch(epoch)
            if train_samplers[ion] is not None:
                train_samplers[ion].set_epoch(epoch)

        for ion in ion_list:
            # Train
            train_metrics = trainer.train_epoch(train_loaders[ion], optimizer, ion)

            # Validate
            val_metrics = trainer.validate_epoch(
                val_loaders[ion], ion, fold_idx=fold_idx
            )

            if is_main_process():
                logging.info(
                    f"Epoch {epoch}/{num_epochs} [{ion}] - "
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"(UCL: {train_metrics['loss_contrastive']:.4f}, "
                    f"Proto: {train_metrics['loss_prototype']:.4f}, "
                    f"BCE: {train_metrics['loss_bce']:.4f}, "
                    f"MLM: {train_metrics['loss_mlm']:.4f})"
                )
                logging.info(
                    f"  Train HRA: {train_metrics['high_recall_auprc']:.3f}, AUPRC: {train_metrics['auprc']:.3f}"
                )
                logging.info(
                    f"  Val HRA: {val_metrics['high_recall_auprc']:.3f}, "
                    f"AUPRC: {val_metrics['auprc']:.3f}, "
                    f"F1: {val_metrics['f1']:.3f}, "
                    f"MCC: {val_metrics['mcc']:.3f}"
                )

            total_train_hra += train_metrics["high_recall_auprc"]
            total_val_hra += val_metrics["high_recall_auprc"]
            total_val_auprc += val_metrics["auprc"]
            val_metrics_per_ion[ion] = val_metrics

        avg_val_hra = total_val_hra / len(ion_list)
        avg_val_auprc = total_val_auprc / len(ion_list)

        # Save best model
        save_model = getattr(conf.training, "save_model", True)
        if avg_val_hra > best_total_hra:
            best_total_hra = avg_val_hra
            best_total_auprc = avg_val_auprc

            best_epoch_metrics = {}
            for ion in ion_list:
                best_epoch_metrics[ion] = val_metrics_per_ion[ion].copy()

            if trainer.is_main_process:
                trainer.snapshot_current_oof_as_best()

            if save_model and is_main_process():
                fold_output_path = f"{base_output_path}/fold_{fold_idx}"
                Path(fold_output_path).mkdir(parents=True, exist_ok=True)
                best_model_path = f"{fold_output_path}/best_e1_contrastive_model.pt"

                trainer.save_checkpoint(
                    path=best_model_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    metrics={"auprc": avg_val_auprc, "hra": avg_val_hra},
                    additional_state={
                        "ions": ion_list,
                        "best_epoch_metrics": best_epoch_metrics,
                        "fold": fold_idx,
                    },
                )
                logging.info(
                    f"New best model at epoch {epoch} with HRA: {best_total_hra:.3f}"
                )

        scheduler.step()

        # Early stopping
        if is_distributed():
            hra_tensor = torch.tensor([avg_val_hra], device=device)
            dist.broadcast(hra_tensor, src=0)
            avg_val_hra = hra_tensor.item()

        should_stop = early_stopper.early_stop(avg_val_hra)

        if is_distributed():
            stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

        if should_stop:
            if is_main_process():
                logging.info(f"Early stopping at epoch {epoch}")
            break

    # Plot training curves
    if is_main_process():
        fold_plots_dir = f"{base_output_path}/fold_{fold_idx}/plots"
        trainer.plot_training_curves(fold=fold_idx, save_dir=fold_plots_dir)

    return {
        "fold": fold_idx,
        "best_epoch_metrics": best_epoch_metrics,
        "best_total_auprc": best_total_auprc,
        "best_total_hra": best_total_hra,
        "model_path": best_model_path,
        "oof_predictions": trainer.get_best_oof(),
    }


def main(conf):
    """Main function."""
    base_output_path = str(conf.output_path)
    return run_cv(
        conf, train_single_fold, base_output_path, log_summary_fn=log_cv_summary
    )


def log_cv_summary(all_fold_results: list, conf):
    """Log cross-validation summary."""
    logging.info(f"\n{'=' * 120}")
    logging.info("K-FOLD CROSS-VALIDATION SUMMARY")
    logging.info(f"{'=' * 120}")

    ion_list = conf.training.ions

    for fold_result in all_fold_results:
        logging.info(f"\nFold {fold_result['fold']}:")
        best_metrics = fold_result["best_epoch_metrics"]

        for ion in ion_list:
            if best_metrics and ion in best_metrics:
                m = best_metrics[ion]
                logging.info(
                    f"  {ion}: AUPRC={m['auprc']:.3f}, HRA={m['high_recall_auprc']:.3f}, "
                    f"F1={m['f1']:.3f}, MCC={m['mcc']:.3f}"
                )

    overall_mean = np.mean([r["best_total_auprc"] for r in all_fold_results])
    overall_std = np.std([r["best_total_auprc"] for r in all_fold_results])
    logging.info(f"\nOverall AUPRC: {overall_mean:.3f} ± {overall_std:.3f}")


if __name__ == "__main__":
    start = timer()

    parser = argparse.ArgumentParser(
        description="E1 contrastive learning for metal ion binding classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="finetune/configs/e1_contrastive.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    local_rank = setup_distributed()

    config_path = Path(args.config)
    config_name = config_path.stem

    conf, output_path = process_config(args.config, config_name=config_name)

    if is_main_process():
        setup_logging(str(output_path))
        logging.info("=" * 80)
        logging.info("E1 Contrastive Learning Fine-tuning")
        logging.info("=" * 80)
        logging.info(f"Config: {args.config}")
        logging.info(f"Output: {output_path}")

    try:
        main(conf)
    finally:
        cleanup_distributed()

    end = timer()
    if is_main_process():
        logging.info(f"Total time: {(end - start) / 60:.1f} minutes")
