import os
import sys

# Add src directory to Python path so E1 module can be imported without installing the package
# This script is in finetune/, so we need to go up one level to reach src/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import logging
import torch
import numpy as np
import shutil
import glob
from typing import List, Optional
from datasets import Dataset

from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from training.e1_dataset import create_e1_datasets_from_config, ConcatE1MSADataset
from training.e1_data_collator import E1DataCollatorForMLM
from training.e1_finetune_utils import load_e1_model
from training.finetune_utils import (
    set_seeds,
    setup_logging,
    save_model,
    ClearCacheCallback,
    MetricRenameCallback,
    process_config,
    MSADatasetEpochCallback,
    CompileFlexAttentionCallback,
)

# Module-level logger (will be configured in train function)
logger = logging.getLogger(__name__)


def create_compute_metrics(
    pad_token_id: int,
    sources: Optional[List[str]] = None,
    batch_preparer=None,
    mlm_probability: float = 0.15,
):
    """
    Create a compute_metrics function for E1 model.

    Args:
        pad_token_id: Padding token ID used by E1 (typically 0)
        sources: List of source labels for each sample (e.g., ["Homologs", "SwissProt", ...])
        batch_preparer: E1BatchPreparer instance for boundary token exclusion

    Returns:
        A compute_metrics function that computes metrics overall and per source
    """

    def compute_metrics(eval_pred):
        """
        Compute perplexity and accuracy for validation.
        If sources are provided, computes metrics separately for each source.

        IMPORTANT: E1 uses pad_token_id (typically 0) for ignored positions, not -100.
        """
        # Handle EvalPrediction object from Transformers
        # EvalPrediction has attributes: predictions, label_ids, and optionally inputs
        if hasattr(eval_pred, "predictions"):
            # New format: EvalPrediction object
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
            # Try to get input_ids and sequence_ids from inputs if available (via include_for_metrics)
            input_ids = None
            sequence_ids = None
            if hasattr(eval_pred, "inputs") and eval_pred.inputs is not None:
                # inputs can be a dict, list of dicts, or tensor
                if isinstance(eval_pred.inputs, dict):
                    input_ids = eval_pred.inputs.get("input_ids", None)
                    sequence_ids = eval_pred.inputs.get("sequence_ids", None)
                elif (
                    isinstance(eval_pred.inputs, (list, tuple))
                    and len(eval_pred.inputs) > 0
                ):
                    # If inputs is a list/tuple of dicts (one per sample)
                    if isinstance(eval_pred.inputs[0], dict):
                        # Stack input_ids and sequence_ids from all samples
                        input_ids_list = [
                            inp.get("input_ids", None)
                            for inp in eval_pred.inputs
                            if isinstance(inp, dict)
                        ]
                        sequence_ids_list = [
                            inp.get("sequence_ids", None)
                            for inp in eval_pred.inputs
                            if isinstance(inp, dict)
                        ]
                        if input_ids_list and all(
                            x is not None for x in input_ids_list
                        ):
                            input_ids = input_ids_list
                        if sequence_ids_list and all(
                            x is not None for x in sequence_ids_list
                        ):
                            sequence_ids = sequence_ids_list
                    elif isinstance(eval_pred.inputs[0], torch.Tensor):
                        # If inputs is a list of tensors, assume first is input_ids
                        input_ids = (
                            eval_pred.inputs[0] if len(eval_pred.inputs) > 0 else None
                        )
                        sequence_ids = (
                            eval_pred.inputs[1] if len(eval_pred.inputs) > 1 else None
                        )
                elif isinstance(eval_pred.inputs, torch.Tensor):
                    # Direct tensor - assume it's input_ids
                    input_ids = eval_pred.inputs
        else:
            # Fallback: Handle tuple format for backward compatibility
            if len(eval_pred) == 3:
                predictions, labels, inputs = eval_pred
                if isinstance(inputs, dict):
                    input_ids = inputs.get("input_ids", None)
                elif isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                    if isinstance(inputs[0], dict):
                        input_ids = [
                            inp.get("input_ids", None)
                            for inp in inputs
                            if isinstance(inp, dict)
                        ]
                    else:
                        input_ids = (
                            inputs[0] if isinstance(inputs[0], torch.Tensor) else None
                        )
                else:
                    input_ids = inputs if isinstance(inputs, torch.Tensor) else None
            elif len(eval_pred) == 2:
                predictions, labels = eval_pred
                input_ids = None
                sequence_ids = None
            else:
                raise ValueError(
                    f"Expected eval_pred to be EvalPrediction object or tuple with 2-3 elements, got {type(eval_pred)}"
                )

        # Extract logits if it's a tuple
        if isinstance(predictions, tuple):
            logits = predictions[0]
        else:
            logits = predictions

        logits_tensor = torch.tensor(logits, dtype=torch.float32)

        # IMPORTANT: Transformers converts pad_token_id (0) to -100 for loss computation
        # But E1 uses pad_token_id = 0, not -100. We MUST convert -100 back to 0 BEFORE filtering!
        # Otherwise, we'll include all ignored positions (since -100 != 0)
        if isinstance(labels, np.ndarray):
            labels_tensor = torch.from_numpy(labels).long()
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Convert -100 (Transformers' ignore_index) back to pad_token_id (0) for E1
        # CRITICAL: This must happen BEFORE filtering, otherwise we'll include all ignored positions!
        num_neg100_before = (
            (labels_tensor == -100).sum().item() if labels_tensor.numel() > 0 else 0
        )
        num_pad_before = (
            (labels_tensor == pad_token_id).sum().item()
            if labels_tensor.numel() > 0
            else 0
        )

        if num_neg100_before > 0:
            logger.info(
                f"Converting -100 to pad_token_id: Found {num_neg100_before} positions with -100, "
                f"{num_pad_before} positions with pad_token_id ({pad_token_id})"
            )
            labels_tensor = torch.where(
                labels_tensor == -100,
                torch.tensor(pad_token_id, dtype=labels_tensor.dtype),
                labels_tensor,
            )
            num_neg100_after = (labels_tensor == -100).sum().item()
            num_pad_after = (labels_tensor == pad_token_id).sum().item()
            logger.info(
                f"After conversion: {num_neg100_after} positions with -100, "
                f"{num_pad_after} positions with pad_token_id ({pad_token_id})"
            )

        # Get input_ids and sequence_ids tensors if available for filtering
        input_ids_tensor = None
        sequence_ids_tensor = None

        if input_ids is not None:
            if isinstance(input_ids, list):
                # If input_ids is a list, stack them into a tensor
                # Handle both list of tensors and list of arrays
                try:
                    if len(input_ids) > 0:
                        if isinstance(input_ids[0], torch.Tensor):
                            input_ids_tensor = torch.stack(input_ids)
                        else:
                            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
                except Exception as e:
                    logger.debug(f"Could not convert input_ids list to tensor: {e}")
                    input_ids_tensor = None
            elif isinstance(input_ids, torch.Tensor):
                input_ids_tensor = input_ids
            else:
                try:
                    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
                except Exception as e:
                    logger.debug(f"Could not convert input_ids to tensor: {e}")
                    input_ids_tensor = None

        if sequence_ids is not None:
            if isinstance(sequence_ids, list):
                try:
                    if len(sequence_ids) > 0:
                        if isinstance(sequence_ids[0], torch.Tensor):
                            sequence_ids_tensor = torch.stack(sequence_ids)
                        else:
                            sequence_ids_tensor = torch.tensor(
                                sequence_ids, dtype=torch.long
                            )
                except Exception as e:
                    logger.debug(f"Could not convert sequence_ids list to tensor: {e}")
                    sequence_ids_tensor = None
            elif isinstance(sequence_ids, torch.Tensor):
                sequence_ids_tensor = sequence_ids
            else:
                try:
                    sequence_ids_tensor = torch.tensor(sequence_ids, dtype=torch.long)
                except Exception as e:
                    logger.debug(f"Could not convert sequence_ids to tensor: {e}")
                    sequence_ids_tensor = None

        # Debug: Log shapes for troubleshooting
        logger.debug(
            f"compute_metrics: logits shape = {logits_tensor.shape}, labels shape = {labels_tensor.shape}"
        )
        logger.debug(
            f"input_ids available: {input_ids_tensor is not None}, sequence_ids available: {sequence_ids_tensor is not None}, batch_preparer available: {batch_preparer is not None}"
        )
        if input_ids_tensor is not None:
            logger.debug(f"input_ids shape: {input_ids_tensor.shape}")
        if sequence_ids_tensor is not None:
            logger.debug(f"sequence_ids shape: {sequence_ids_tensor.shape}")

        # Handle different shapes that Trainer might provide
        # During evaluation, predictions might be already flattened
        if len(logits_tensor.shape) == 1:
            # This case happens when logits are completely collapsed
            # Usually means vocab_size only - indicates a problem with prediction extraction
            if (
                logits_tensor.shape[0] <= 512
            ):  # Likely just vocab size (E1 has 376 tokens)
                raise ValueError(
                    f"Logits appear to be just vocab size ({logits_tensor.shape[0]}). "
                    f"This indicates the model output is not being extracted correctly. "
                    f"Labels shape: {labels_tensor.shape}. "
                    f"Check preprocess_logits_for_metrics function."
                )
            # If it's longer, might be flattened predictions
            if len(labels_tensor.shape) == 1:
                # Both are flat - we can't reconstruct batch/seq structure
                labels_flat = labels_tensor
                # This shouldn't happen - logits need vocab dimension
                raise ValueError(
                    f"Cannot handle completely flattened predictions. "
                    f"Logits shape: {logits_tensor.shape}, Labels shape: {labels_tensor.shape}"
                )
            else:
                raise ValueError(
                    f"Unexpected shape combination: logits {logits_tensor.shape}, labels {labels_tensor.shape}"
                )
        elif len(logits_tensor.shape) == 2:
            # Logits are (total_tokens, vocab_size) - already flattened across batch and seq
            # Labels should be (total_tokens,) or (batch_size, seq_len)
            if len(labels_tensor.shape) == 1:
                # Both flattened to token level
                logits_flat = logits_tensor
                labels_flat = labels_tensor
            elif len(labels_tensor.shape) == 2:
                # Labels still have batch/seq structure, flatten them
                labels_flat = labels_tensor.view(-1)
                logits_flat = logits_tensor
                # Verify sizes match
                if logits_flat.shape[0] != labels_flat.shape[0]:
                    raise ValueError(
                        f"Size mismatch after flattening: logits {logits_flat.shape[0]} vs labels {labels_flat.shape[0]}"
                    )
            else:
                raise ValueError(f"Unexpected labels shape: {labels_tensor.shape}")
        elif len(logits_tensor.shape) == 3:
            # Standard case: (batch_size, seq_len, vocab_size)
            if len(labels_tensor.shape) != 2:
                raise ValueError(
                    f"Expected labels to be 2D when logits are 3D, got shape {labels_tensor.shape}"
                )

            batch_size, seq_len, vocab_size = logits_tensor.shape

            # Verify batch and seq dimensions match
            if (
                labels_tensor.shape[0] != batch_size
                or labels_tensor.shape[1] != seq_len
            ):
                raise ValueError(
                    f"Shape mismatch: logits {logits_tensor.shape} vs labels {labels_tensor.shape}"
                )

            # Reshape to (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
            logits_flat = logits_tensor.view(-1, vocab_size)
            labels_flat = labels_tensor.view(-1)
        else:
            raise ValueError(f"Unexpected logits shape: {logits_tensor.shape}")

        # Create mask for valid labels (not pad_token_id, which means ignore this position)
        # E1 uses pad_token_id instead of -100
        # IMPORTANT: The data collator (E1DataCollatorForMLM) already:
        # 1. Only masks positions in query sequence (selected_mask is subset of query_mask)
        # 2. Excludes boundary tokens (selected_mask is subset of valid_mask = query_mask & (~boundary_mask))
        # 3. Sets labels only for selected_mask positions (all others are pad_token_id)
        # So labels != pad_token_id should already give us the correct positions!
        # However, we still try to apply additional filtering if inputs are available for extra safety
        mask = labels_flat != pad_token_id

        # Log initial mask stats for debugging
        num_masked_positions = mask.sum().item()
        total_positions = len(mask)
        if num_masked_positions == 0:
            raise ValueError(
                "No masked positions found! This indicates a problem with the data collator."
            )

        # Sanity check: masked positions should be a small percentage
        # Note: The ratio is low because:
        # - MLM probability (~15%) applies only to query sequence positions
        # - Query sequence is only part of total (context + query sequences)
        # - With MSA context (max_num_samples=64-256), query can be 5-15% of total
        # - So ~0.75-2.25% of total positions = ~15% of query sequence is expected
        masked_ratio = (
            num_masked_positions / total_positions if total_positions > 0 else 0
        )
        # With MSA context, query is typically 5-15% of total tokens
        # (depends on max_num_samples and max_token_length config)
        expected_ratio_low = (
            mlm_probability * 0.05
        )  # ~0.75% if query is ~5% of total (many context seqs)
        expected_ratio_high = (
            mlm_probability * 0.30
        )  # ~4.5% if query is ~30% of total (few context seqs)

        logger.info(
            f"Masked positions: {num_masked_positions}/{total_positions} ({100*masked_ratio:.2f}%). "
            f"Expected ~{mlm_probability*100:.1f}% of query sequence "
            f"(~{100*expected_ratio_low:.1f}-{100*expected_ratio_high:.1f}% of total positions, "
            f"depending on MSA context size)."
        )

        # If masked ratio is suspiciously high, it might indicate we're including context sequences
        if masked_ratio > 0.5:
            logger.warning(
                f"Suspiciously high masked ratio ({100*masked_ratio:.2f}%). "
                f"This might indicate context sequence positions are being included."
            )
        elif masked_ratio < expected_ratio_low * 0.5:
            # Only warn if extremely low (< 0.375% with 15% MLM prob)
            # This could indicate empty/very short query sequences
            logger.warning(
                f"Very low masked ratio ({100*masked_ratio:.2f}%). "
                f"This may indicate very long context sequences or short query sequences. "
                f"Expected at least ~{100*expected_ratio_low:.1f}% of total positions."
            )

        # Match evaluation script: create valid_query_mask = query_mask & (~boundary_mask)
        # Check original shape before flattening (we need 3D shape for batch structure)
        has_batch_structure = len(logits_tensor.shape) == 3

        if (
            sequence_ids_tensor is not None
            and input_ids_tensor is not None
            and batch_preparer is not None
            and has_batch_structure
        ):
            # We have batch structure, create valid_query_mask exactly like evaluation script
            batch_size, seq_len, vocab_size = logits_tensor.shape

            # Flatten input_ids and sequence_ids to match labels_flat shape
            if len(input_ids_tensor.shape) > 1:
                input_ids_flat = input_ids_tensor.view(-1)
            else:
                input_ids_flat = input_ids_tensor
            sequence_ids_flat = sequence_ids_tensor.view(-1)

            # Verify shapes match
            if (
                input_ids_flat.shape[0] != labels_flat.shape[0]
                or sequence_ids_flat.shape[0] != labels_flat.shape[0]
            ):
                logger.warning(
                    f"Shape mismatch: input_ids_flat {input_ids_flat.shape[0]}, "
                    f"sequence_ids_flat {sequence_ids_flat.shape[0]} vs labels_flat {labels_flat.shape[0]}. "
                    f"Skipping query sequence and boundary token filtering."
                )
            else:
                # Step 1: Create query_mask (positions in query sequence = max sequence_id)
                query_mask = torch.zeros_like(sequence_ids_flat, dtype=torch.bool)
                for i in range(batch_size):
                    sample_seq_ids = sequence_ids_tensor[i]
                    max_seq_id = sample_seq_ids.max().item()
                    query_mask[i * seq_len : (i + 1) * seq_len] = (
                        sample_seq_ids == max_seq_id
                    )

                # Step 2: Create boundary_mask (exclude boundary tokens)
                boundary_token_ids = batch_preparer.boundary_token_ids.to(
                    input_ids_flat.device
                )
                boundary_mask = torch.isin(input_ids_flat, boundary_token_ids)

                # Step 3: Combine like evaluation script: valid_query_mask = query_mask & (~boundary_mask)
                valid_query_mask = query_mask & (~boundary_mask)

                # Step 4: Apply valid_query_mask to our mask (which already filters to masked positions)
                mask = mask & valid_query_mask

                # Log for debugging
                num_query = query_mask.sum().item()
                num_boundary = boundary_mask.sum().item()
                num_valid_query = valid_query_mask.sum().item()
                num_masked_before = (labels_flat != pad_token_id).sum().item()
                num_masked_after = mask.sum().item()
                logger.info(
                    f"Filtering (matching eval script): {num_query} query positions, "
                    f"{num_boundary} boundary tokens, {num_valid_query} valid query positions, "
                    f"{num_masked_before} masked positions before filtering -> {num_masked_after} after"
                )
        elif sequence_ids_tensor is not None and has_batch_structure:
            # Fallback: only filter to query sequence if we don't have input_ids
            batch_size, seq_len, vocab_size = logits_tensor.shape
            sequence_ids_flat = sequence_ids_tensor.view(-1)

            if sequence_ids_flat.shape[0] == labels_flat.shape[0]:
                query_mask = torch.zeros_like(sequence_ids_flat, dtype=torch.bool)
                for i in range(batch_size):
                    sample_seq_ids = sequence_ids_tensor[i]
                    max_seq_id = sample_seq_ids.max().item()
                    query_mask[i * seq_len : (i + 1) * seq_len] = (
                        sample_seq_ids == max_seq_id
                    )
                mask = mask & query_mask
                logger.warning(
                    "Only query sequence filtering applied (no input_ids for boundary exclusion)"
                )
        else:
            # NOTE: include_for_metrics doesn't pass inputs through to compute_metrics
            # However, this is OK because the data collator (E1DataCollatorForMLM) already:
            # 1. Only masks positions in query sequence (selected_mask ⊆ query_mask)
            # 2. Excludes boundary tokens (selected_mask ⊆ valid_mask = query_mask & (~boundary_mask))
            # 3. Sets labels only for selected_mask positions (all others are pad_token_id)
            # So labels != pad_token_id already gives us the correct positions!
            # The conversion from -100 to pad_token_id ensures we filter correctly.
            logger.debug(
                f"Additional filtering not applied (inputs not available via include_for_metrics). "
                f"This is OK - data collator already filters correctly. "
                f"sequence_ids={sequence_ids_tensor is not None}, "
                f"input_ids={input_ids_tensor is not None}"
            )

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

        # Debug logging with more details
        logger.info(
            f"Accuracy calculation: {correct_predictions}/{total_predictions} = {accuracy:.4f} "
            f"({100*accuracy:.2f}%), perplexity = {perplexity:.4f}, loss = {average_loss:.4f}"
        )

        # Additional sanity check: if accuracy is suspiciously low, log more details
        if accuracy < 0.1 and total_predictions > 100:
            # Sample some predictions to see what's happening
            sample_size = min(10, total_predictions)
            sample_indices = torch.randperm(total_predictions)[:sample_size]
            sample_pred = predicted_tokens[sample_indices]
            sample_labels = masked_labels[sample_indices]
            logger.warning(
                f"Low accuracy detected. Sample predictions vs labels: "
                f"pred={sample_pred.tolist()}, labels={sample_labels.tolist()}, "
                f"matches={(sample_pred == sample_labels).sum().item()}/{sample_size}"
            )

        metrics = {"perplexity": perplexity, "accuracy": accuracy}

        if total_predictions == 0:
            raise ValueError("No masked positions found for metric computation!")

        # Compute separate metrics for each source if sources are provided
        # Only possible when we have 3D logits (batch structure preserved)
        if sources is not None and len(logits_tensor.shape) == 3:
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

                        # Exclude boundary tokens if input_ids is available
                        if input_ids_tensor is not None and batch_preparer is not None:
                            sample_input_ids = input_ids_tensor[idx]
                            boundary_token_ids = batch_preparer.boundary_token_ids.to(
                                sample_input_ids.device
                            )
                            boundary_mask = torch.isin(
                                sample_input_ids, boundary_token_ids
                            )
                            sample_mask = sample_mask & (~boundary_mask)

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

    # Get validation MSA sampling config (for fair comparison with evaluation)
    validation_msa_sampling_conf = train_conf.get("validation_msa_sampling", {})

    # Create datasets using E1MSADataset
    homologs_train, homologs_val, swissprot_train, swissprot_val = (
        create_e1_datasets_from_config(
            data_conf=data_conf,
            general_conf=general_conf,
            msa_sampling_conf=msa_sampling_conf,
            validation_msa_sampling_conf=validation_msa_sampling_conf,
        )
    )

    logger.info(f"Homologs: Train {len(homologs_train)}, Val {len(homologs_val)}")
    logger.info(f"SwissProt: Train {len(swissprot_train)}, Val {len(swissprot_val)}")

    # Create dynamic training dataset using ConcatE1MSADataset
    # This allows different MSA samples each epoch (data augmentation)
    train_datasets = []
    train_sources = []

    if len(homologs_train) > 0:
        train_datasets.append(homologs_train)
        train_sources.append("Homologs")

    if len(swissprot_train) > 0:
        train_datasets.append(swissprot_train)
        train_sources.append("SwissProt")

    # ConcatE1MSADataset defers MSA sampling to __getitem__ time
    # enabling different context samples per epoch
    train_set = ConcatE1MSADataset(
        datasets=train_datasets,
        sources=train_sources,
        shuffle=True,
        seed=general_conf["seed"],
    )

    logger.info(f"Created dynamic training dataset with {len(train_set)} total samples")
    logger.info("  MSA sampling happens at access time (different samples per epoch)")

    # Prepare validation data
    # Validation uses static extraction since we want consistent evaluation
    val_sequences = []
    val_sources = []

    # Add homolog validation sequences (sampled with validation MSA params)
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
    # Get max_token_length and max_query_length from training and validation MSA sampling configs
    # Use the maximum to ensure validation sequences aren't truncated unnecessarily
    msa_sampling_conf = train_conf.get("msa_sampling", {})
    validation_msa_sampling_conf = train_conf.get("validation_msa_sampling", {})
    train_max_token_length = msa_sampling_conf.get("max_token_length", 8192)
    train_max_query_length = msa_sampling_conf.get("max_query_length", 1024)
    val_max_token_length = validation_msa_sampling_conf.get(
        "max_token_length", train_max_token_length
    )
    val_max_query_length = validation_msa_sampling_conf.get(
        "max_query_length", train_max_query_length
    )
    # Use maximum of training and validation limits for data collator
    max_token_length = max(train_max_token_length, val_max_token_length)
    max_query_length = max(train_max_query_length, val_max_query_length)
    data_collator = E1DataCollatorForMLM(
        mlm_probability=mlm_probability,
        max_total_tokens=max_token_length,  # Cap total tokens to prevent OOM on long MSAs
        max_query_tokens=max_query_length,  # Cap query length (O(n²) self-attention)
    )

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
        include_for_metrics=[
            "input_ids",
            "sequence_ids",
        ],  # Include input_ids and sequence_ids for filtering
    )

    # Create compute_metrics function with source information
    compute_metrics_fn = create_compute_metrics(
        pad_token_id=pad_token_id,
        sources=val_sources,
        batch_preparer=batch_preparer,
        mlm_probability=mlm_probability,
    )

    # Build callbacks list
    callbacks_list = [
        ClearCacheCallback(),
        MetricRenameCallback(),
        MSADatasetEpochCallback(train_set),  # Enable dynamic MSA sampling per epoch
        CompileFlexAttentionCallback(),  # Compile flex_attention for improved training speed
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
    # Handles both dict format (from ConcatE1MSADataset) and HuggingFace Dataset format
    def collate_fn(examples):
        # Extract text strings from examples
        # ConcatE1MSADataset returns {"text": ..., "source": ...}
        # HuggingFace Dataset (validation) returns {"text": ...}
        texts = [ex["text"] for ex in examples]
        batch = data_collator(texts)
        return batch

    # Preprocess logits to ensure they're in the right format for metrics
    def preprocess_logits_for_metrics(logits, labels):
        """
        Preprocess logits before metric computation.
        E1ForMaskedLM returns E1MaskedLMOutputWithPast with 'logits' field.

        Args:
            logits: Model output (tuple or ModelOutput)
            labels: Ground truth labels

        Returns:
            Logits tensor of shape (batch, seq, vocab)
        """
        # Handle different output formats
        if hasattr(logits, "logits"):
            # ModelOutput object with logits attribute
            return logits.logits
        elif isinstance(logits, tuple) and len(logits) > 1:
            # Tuple output: (loss, logits, ...)
            # Return the second element (logits)
            return logits[1]
        elif isinstance(logits, tuple) and len(logits) == 1:
            # Single element tuple
            item = logits[0]
            if hasattr(item, "logits"):
                return item.logits
            else:
                return item
        else:
            # Assume it's already logits tensor
            return logits

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_dict,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks_list,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
    msa_sampling_conf = train_conf.get("msa_sampling", {})
    logger.info(
        f"Max token length (training): {msa_sampling_conf.get('max_token_length', 'N/A')}"
    )
    logger.info(
        f"Max query length (training): {msa_sampling_conf.get('max_query_length', 'N/A')}"
    )
    validation_msa_sampling_conf = train_conf.get("validation_msa_sampling", {})
    logger.info(
        f"Max token length (validation): {validation_msa_sampling_conf.get('max_token_length', 'N/A')}"
    )
    logger.info(
        f"Max query length (validation): {validation_msa_sampling_conf.get('max_query_length', 'N/A')}"
    )
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
    from pathlib import Path

    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run E1 LoRA Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="finetune/configs/e1_lora_config.yaml",
        help="Path to config YAML",
    )

    args = parser.parse_args()

    # Extract config name from config path for directory structuring
    config_path = Path(args.config)
    config_name = config_path.stem  # e.g., "e1_lora_config" from "e1_lora_config.yaml"

    # Load config and setup output path using process_config
    config, output_path = process_config(args.config, config_name=config_name)

    train(config, output_path=output_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Total time used: {(time.time() - start_time)/60:.1f} minutes")
