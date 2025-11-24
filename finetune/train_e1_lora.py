import os
import argparse
import logging
import torch
import numpy as np
import shutil
import glob
import yaml
from typing import List, Optional
from datasets import Dataset

from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from training.e1_dataset import create_e1_datasets_from_config
from training.e1_data_collator import E1DataCollatorForMLM
from training.e1_finetune_utils import load_e1_model
from training.finetune_utils import (
    set_seeds,
    setup_logging,
    save_model,
    ClearCacheCallback,
    MetricRenameCallback,
)

# Module-level logger (will be configured in train function)
logger = logging.getLogger(__name__)


def create_compute_metrics(pad_token_id: int, sources: Optional[List[str]] = None):
    """
    Create a compute_metrics function for E1 model.

    Args:
        pad_token_id: Padding token ID used by E1 (typically 0)
        sources: List of source labels for each sample (e.g., ["Homologs", "SwissProt", ...])

    Returns:
        A compute_metrics function that computes metrics overall and per source
    """

    def compute_metrics(eval_pred):
        """
        Compute perplexity and accuracy for validation.
        If sources are provided, computes metrics separately for each source.

        IMPORTANT: E1 uses pad_token_id (typically 0) for ignored positions, not -100.
        """
        predictions, labels = eval_pred

        # Extract logits if it's a tuple
        if isinstance(predictions, tuple):
            logits = predictions[0]
        else:
            logits = predictions

        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Verify shapes
        if len(logits_tensor.shape) != 3:
            raise ValueError(
                f"Expected logits to be 3D, got shape {logits_tensor.shape}"
            )
        if len(labels_tensor.shape) != 2:
            raise ValueError(
                f"Expected labels to be 2D, got shape {labels_tensor.shape}"
            )

        batch_size, seq_len, vocab_size = logits_tensor.shape

        # Verify batch and seq dimensions match
        if labels_tensor.shape[0] != batch_size or labels_tensor.shape[1] != seq_len:
            raise ValueError(
                f"Shape mismatch: logits {logits_tensor.shape} vs labels {labels_tensor.shape}"
            )

        # Reshape to (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
        logits_flat = logits_tensor.view(-1, vocab_size)
        labels_flat = labels_tensor.view(-1)

        # Create mask for valid labels (not pad_token_id, which means ignore this position)
        # E1 uses pad_token_id instead of -100
        mask = labels_flat != pad_token_id

        # Check if we have any masked positions
        if mask.sum() == 0:
            raise ValueError(
                "No masked positions found! Check data collator configuration."
            )

        # Apply mask to get only masked positions (where labels != pad_token_id)
        masked_logits = logits_flat[mask]  # Shape: (num_masked, vocab_size)
        masked_labels = labels_flat[mask]  # Shape: (num_masked,)

        # Compute loss on masked positions only
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(masked_logits, masked_labels)
        average_loss = loss.item()
        perplexity = np.exp(average_loss)

        # Compute accuracy on masked positions
        predicted_tokens = torch.argmax(masked_logits, dim=-1)
        correct_predictions = (predicted_tokens == masked_labels).sum().item()
        total_predictions = masked_labels.size(0)
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0.0
        )

        metrics = {"perplexity": perplexity, "accuracy": accuracy}

        if total_predictions == 0:
            raise ValueError("No masked positions found for metric computation!")

        # Compute separate metrics for each source if sources are provided
        if sources is not None:
            num_samples = labels_tensor.shape[0]

            # Compute metrics per source by iterating through samples
            unique_sources = list(set(sources))
            for source in unique_sources:
                source_indices = [i for i, s in enumerate(sources) if s == source]

                # Collect tokens from samples belonging to this source
                source_masked_logits_list = []
                source_masked_labels_list = []

                for idx in source_indices:
                    if idx < num_samples:
                        sample_mask = labels_tensor[idx] != pad_token_id
                        if sample_mask.any():
                            source_masked_logits_list.append(
                                logits_tensor[idx][sample_mask]
                            )
                            source_masked_labels_list.append(
                                labels_tensor[idx][sample_mask]
                            )

                if len(source_masked_logits_list) > 0:
                    source_logits = torch.cat(source_masked_logits_list, dim=0)
                    source_labels = torch.cat(source_masked_labels_list, dim=0)

                    source_loss = loss_fn(source_logits, source_labels)
                    source_perplexity = np.exp(source_loss.item())

                    _, source_predicted = torch.max(source_logits, dim=-1)
                    source_correct = (source_predicted == source_labels).sum().item()
                    source_accuracy = source_correct / source_labels.size(0)

                    metrics[f"{source}_perplexity"] = source_perplexity
                    metrics[f"{source}_accuracy"] = source_accuracy

        return metrics

    return compute_metrics


def train(config, output_path=None):
    # Extract config
    general_conf = config["general"]
    data_conf = config["data"]
    train_conf = config["training"]
    lora_conf = config["lora"]
    msa_sampling_conf = train_conf.get("msa_sampling", {})

    set_seeds(general_conf["seed"])

    # Determine output directory
    if output_path is not None:
        output_dir = str(output_path)
    else:
        output_dir = train_conf.get("output_dir", "results/e1_lora_training")

    # Create output directory early for logging
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging (must be done after output_dir is created)
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Starting E1 LoRA Fine-tuning")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")

    # Check for resume from checkpoint
    resume_from_checkpoint = train_conf.get("resume_from_checkpoint", None)
    save_best_checkpoint = train_conf.get("save_best_checkpoint", True)

    if resume_from_checkpoint is not None:
        if not os.path.exists(resume_from_checkpoint):
            raise ValueError(f"Resume checkpoint not found: {resume_from_checkpoint}")
        if not os.path.isdir(resume_from_checkpoint):
            raise ValueError(
                f"Resume checkpoint must be a directory: {resume_from_checkpoint}"
            )

        # Check for essential checkpoint files (LoRA-specific)
        required_files = [
            "trainer_state.json",
            "optimizer.pt",
            "scaler.pt",
            "scheduler.pt",
            "adapter_model.safetensors",
        ]
        missing_files = [
            f
            for f in required_files
            if not os.path.exists(os.path.join(resume_from_checkpoint, f))
        ]
        if missing_files:
            raise ValueError(
                f"Invalid checkpoint - missing files: {missing_files} in {resume_from_checkpoint}"
            )

        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
    else:
        logger.info("Starting fresh training (no resume checkpoint specified)")

    logger.info(f"Save best checkpoint: {save_best_checkpoint}")

    # Check if we're in distributed training mode (set by torchrun)
    is_distributed = os.environ.get("RANK") is not None

    # Handle GPU selection
    if not is_distributed:
        gpu_id = general_conf.get("gpu_id", "0")
        if isinstance(gpu_id, int):
            gpu_id = str(gpu_id)
        elif isinstance(gpu_id, list):
            gpu_id = ",".join(map(str, gpu_id))

        if gpu_id.lower() != "all":
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            num_gpus = len(gpu_id.split(","))
            logger.info(f"Using GPU(s): {gpu_id}")
        else:
            num_gpus = torch.cuda.device_count()
            logger.info(f"Using all available GPUs: {num_gpus}")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        logger.info(
            f"Distributed training (DDP): RANK={os.environ.get('RANK')}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}"
        )

    logger.info("Loading data...")

    # Create datasets using E1MSADataset
    homologs_train, homologs_val, swissprot_train, swissprot_val = (
        create_e1_datasets_from_config(
            data_conf=data_conf,
            general_conf=general_conf,
            msa_sampling_conf=msa_sampling_conf,
        )
    )

    logger.info(f"Homologs: Train {len(homologs_train)}, Val {len(homologs_val)}")
    logger.info(f"SwissProt: Train {len(swissprot_train)}, Val {len(swissprot_val)}")

    # Combine training datasets
    # For E1, we need to combine the datasets properly
    # Since E1MSADataset returns strings, we can combine them
    train_sequences = []
    train_sources = []

    # Add homolog sequences
    for i in range(len(homologs_train)):
        train_sequences.append(homologs_train[i])
        train_sources.append("Homologs")

    # Add SwissProt sequences
    for i in range(len(swissprot_train)):
        train_sequences.append(swissprot_train[i])
        train_sources.append("SwissProt")

    # Shuffle
    import random

    random.seed(general_conf["seed"])
    combined = list(zip(train_sequences, train_sources))
    random.shuffle(combined)
    train_sequences, train_sources = zip(*combined)
    train_sequences = list(train_sequences)
    train_sources = list(train_sources)

    # Create HuggingFace Dataset from list of strings
    train_set = Dataset.from_dict({"text": train_sequences})

    # Prepare validation data
    val_sequences = []
    val_sources = []

    # Add homolog validation sequences
    for i in range(len(homologs_val)):
        val_sequences.append(homologs_val[i])
        val_sources.append("Homologs")

    # Add SwissProt validation sequences
    for i in range(len(swissprot_val)):
        val_sequences.append(swissprot_val[i])
        val_sources.append("SwissProt")

    # Limit validation sets if needed
    max_eval_samples = train_conf.get("max_eval_samples", None)
    if max_eval_samples is not None and len(val_sequences) > max_eval_samples:
        import random

        random.seed(general_conf["seed"])
        indices = random.sample(range(len(val_sequences)), max_eval_samples)
        val_sequences = [val_sequences[i] for i in indices]
        val_sources = [val_sources[i] for i in indices]
        logger.info(
            f"Limited validation set to {max_eval_samples} samples to prevent OOM"
        )

    val_set = Dataset.from_dict({"text": val_sequences})
    val_dict = {"Combined": val_set}

    # Get model dtype from config
    model_dtype = train_conf.get("model_dtype", None)

    # Load model and batch preparer
    model, batch_preparer = load_e1_model(
        train_conf["checkpoint"], lora_conf, model_dtype=model_dtype
    )

    pad_token_id = batch_preparer.pad_token_id

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)"
    )

    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found! Check LoRA configuration.")

    # Ensure model is in training mode
    model.train()

    # Create data collator
    mlm_probability = train_conf.get("mlm_probability", 0.15)
    data_collator = E1DataCollatorForMLM(mlm_probability=mlm_probability)

    # Mixed Precision Settings
    fp16 = False
    bf16 = False
    if train_conf["mixed_precision"] == "fp16":
        fp16 = True
    elif train_conf["mixed_precision"] == "bf16":
        bf16 = True
    fp16_full_eval = fp16
    bf16_full_eval = bf16

    # Step-based evaluation, logging, and saving settings
    eval_steps = train_conf.get("eval_steps", 50)
    logging_steps = train_conf.get("logging_steps", 10)

    # Evaluation batch size
    eval_batch_size = train_conf.get(
        "eval_batch_size", max(1, train_conf["batch_size"] // 2)
    )

    # eval_accumulation_steps
    eval_accumulation_steps = train_conf.get("eval_accumulation_steps", 10)

    # Get warmup steps from config
    warmup_steps = train_conf.get("warmup_steps", 0)
    lr_scheduler_type = train_conf.get("lr_scheduler_type", "cosine")

    # DDP settings
    ddp_find_unused_parameters = train_conf.get("ddp_find_unused_parameters", False)

    # Early stopping configuration
    early_stopping_enabled = train_conf.get("early_stopping", {}).get("enabled", False)
    early_stopping_patience = train_conf.get("early_stopping", {}).get("patience", 3)
    early_stopping_metric = train_conf.get("early_stopping", {}).get(
        "metric", "eval_Combined_perplexity"
    )
    early_stopping_threshold = train_conf.get("early_stopping", {}).get(
        "threshold", 0.0
    )

    # Best model metric
    best_metric = (
        early_stopping_metric if early_stopping_enabled else "eval_Combined_perplexity"
    )
    greater_is_better = "accuracy" in best_metric.lower()

    # Get weight decay and optimizer from config
    weight_decay = train_conf.get("weight_decay", 0.0)
    optimizer = train_conf.get("optimizer", "adamw_torch_fused")

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_dir=output_dir,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        learning_rate=train_conf["learning_rate"],
        weight_decay=weight_decay,
        optim=optimizer,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        per_device_train_batch_size=train_conf["batch_size"],
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=train_conf["accum_steps"],
        num_train_epochs=train_conf["epochs"],
        seed=general_conf["seed"],
        fp16=fp16,
        bf16=bf16,
        fp16_full_eval=fp16_full_eval,
        bf16_full_eval=bf16_full_eval,
        gradient_checkpointing=train_conf.get("gradient_checkpointing", False),
        report_to=[],
        dataloader_num_workers=train_conf.get("dataloader_num_workers", 0),
        dataloader_pin_memory=False,
        prediction_loss_only=False,
        eval_accumulation_steps=eval_accumulation_steps,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=greater_is_better,
        save_on_each_node=False,
        remove_unused_columns=False,  # Required: Our data collator converts "text" strings to model inputs
    )

    # Create compute_metrics function with source information
    compute_metrics_fn = create_compute_metrics(
        pad_token_id=pad_token_id, sources=val_sources
    )

    # Build callbacks list
    callbacks_list = [
        ClearCacheCallback(),
        MetricRenameCallback(),
    ]

    # Add early stopping callback if enabled
    if early_stopping_enabled:
        callbacks_list.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    # Custom data collator function that extracts text field
    def collate_fn(examples):
        # Extract text strings from examples
        texts = [ex["text"] for ex in examples]
        return data_collator(texts)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_dict,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks_list,
    )

    logger.info("=" * 80)
    logger.info("Training Configuration Summary")
    logger.info("=" * 80)
    logger.info(f"Model: {train_conf['checkpoint']}")
    logger.info(f"Model dtype: {model_dtype if model_dtype else 'float32'}")
    logger.info(f"Epochs: {train_conf['epochs']}")
    logger.info(f"Batch size (per device): {train_conf['batch_size']}")
    logger.info(f"Gradient accumulation steps: {train_conf['accum_steps']}")
    logger.info(
        f"Effective batch size: {train_conf['batch_size'] * train_conf['accum_steps']}"
    )
    logger.info(f"Learning rate: {train_conf['learning_rate']}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"LR scheduler: {lr_scheduler_type} (warmup steps: {warmup_steps})")
    logger.info(f"Mixed precision: {train_conf['mixed_precision']}")
    logger.info(f"Max sequence length: {data_conf.get('max_length', 'N/A')}")
    logger.info(
        f"Gradient checkpointing: {train_conf.get('gradient_checkpointing', False)}"
    )
    logger.info(f"Logging: Every {logging_steps} steps -> {output_dir}")
    logger.info(f"Evaluation: Every {eval_steps} steps")
    logger.info(f"Evaluation batch size: {eval_batch_size}")
    logger.info(f"Max evaluation samples per set: {max_eval_samples}")
    logger.info(
        f"Eval accumulation steps: {eval_accumulation_steps} (moves predictions to CPU periodically)"
    )
    logger.info(f"Training samples: {len(train_set)}")
    logger.info(f"Validation sets: {list(val_dict.keys())}")
    for name, val_set in val_dict.items():
        logger.info(f"  - {name}: {len(val_set)} samples")
    logger.info("=" * 80)
    logger.info("Starting training...")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the best model weights
    best_model_path = os.path.join(output_dir, "best_model.pth")
    save_model(model, best_model_path)
    logger.info(f"Best model weights saved to {best_model_path}")

    # Save the best checkpoint folder if requested
    if save_best_checkpoint:
        best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")

        if (
            hasattr(trainer.state, "best_model_checkpoint")
            and trainer.state.best_model_checkpoint
        ):
            best_checkpoint_source = trainer.state.best_model_checkpoint
            if os.path.exists(best_checkpoint_source):
                if os.path.exists(best_checkpoint_dir):
                    shutil.rmtree(best_checkpoint_dir)
                shutil.copytree(best_checkpoint_source, best_checkpoint_dir)
                logger.info(f"Best checkpoint saved to {best_checkpoint_dir}")
            else:
                logger.warning(
                    f"Best checkpoint source not found: {best_checkpoint_source}"
                )
        else:
            logger.warning("No best checkpoint found in trainer state")

    # Clean up intermediate checkpoints
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoint_dirs = glob.glob(checkpoint_pattern)
    for checkpoint_path in checkpoint_dirs:
        if save_best_checkpoint and checkpoint_path == os.path.join(
            output_dir, "best_checkpoint"
        ):
            continue
        try:
            shutil.rmtree(checkpoint_path)
            logger.info(f"Removed intermediate checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Could not remove checkpoint {checkpoint_path}: {e}")

    logger.info("Training complete.")


if __name__ == "__main__":
    import time

    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run E1 LoRA Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="finetune/configs/e1_lora_config.yaml",
        help="Path to config YAML",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)
    logger = logging.getLogger(__name__)
    logger.info(f"Total time used: {(time.time() - start_time)/60:.1f} minutes")
