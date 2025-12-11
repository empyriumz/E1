import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm

# E1 imports
from E1.batch_preparer import E1BatchPreparer
from E1.modeling import E1ForMaskedLM
from E1.msa_sampling import sample_context

# Module-level logger (will be configured in main function)
logger = logging.getLogger(__name__)


def set_seeds(s: int):
    """Set random seeds for reproducibility."""
    import random

    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def setup_logging(output_dir: str):
    """
    Set up logging to both console and file.

    Args:
        output_dir: Directory where log file will be saved
    """
    log_file = os.path.join(output_dir, "evaluation.log")

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def create_masked_sequence(
    sequence: str,
    mask_token: str,
    mlm_probability: float = 0.15,
) -> str:
    """
    Create a masked sequence for MLM evaluation.

    Args:
        sequence: Original protein sequence
        mask_token: Token to use for masking (e.g., "?" for E1)
        mlm_probability: Probability of masking each character

    Returns:
        Masked sequence string
    """
    import random

    # Convert sequence to list of characters for easier manipulation
    seq_chars = list(sequence)

    # Create masked sequence
    for i, char in enumerate(seq_chars):
        if random.random() < mlm_probability:
            # Mask this position
            seq_chars[i] = mask_token

    masked_sequence = "".join(seq_chars)
    return masked_sequence


def load_original_model(
    model_name: str,
    device: torch.device,
    model_dtype: Optional[str] = None,
):
    """
    Load original E1 model directly from HuggingFace.

    Args:
        model_name: HuggingFace model identifier (e.g., "Profluent-Bio/E1-600m")
        device: Device to load model on
        model_dtype: Model dtype ("float16", "bfloat16", or None for float32)
    """
    logger.info(f"Loading original E1 model from HuggingFace: {model_name}...")

    # Determine dtype
    dtype = None
    if model_dtype == "float16":
        dtype = torch.float16
    elif model_dtype == "bfloat16":
        dtype = torch.bfloat16

    # Load model
    if dtype is not None:
        model = E1ForMaskedLM.from_pretrained(model_name, dtype=dtype).to(device)
    else:
        model = E1ForMaskedLM.from_pretrained(model_name).to(device)

    # Create batch preparer
    batch_preparer = E1BatchPreparer()

    # Log number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Original Model - Total Parameters: {total_params/1e6:.1f}M")

    # Set model to evaluation mode
    model.eval()

    return model, batch_preparer


def evaluate_sequence_batch(
    model: E1ForMaskedLM,
    batch_preparer: E1BatchPreparer,
    sequences: List[Tuple[str, str]],  # List of (seq_id, sequence) tuples
    msa_dir: Optional[str],
    mlm_probability: float,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate a batch of sequences and compute metrics.

    Args:
        model: E1 model
        batch_preparer: E1BatchPreparer instance
        sequences: List of (seq_id, sequence) tuples to evaluate
        msa_dir: Directory containing MSA files (.a3m format)
        mlm_probability: Probability of masking tokens
        device: Device to run evaluation on

    Returns:
        Dictionary with metrics
    """
    all_losses = []
    all_correct = 0
    all_total = 0

    mask_token = batch_preparer.mask_token  # "?" for E1
    mask_token_id = batch_preparer.mask_token_id  # Pre-computed mask token ID
    padding_idx = batch_preparer.pad_token_id

    # Process each sequence with progress bar
    progress_bar = tqdm(sequences, total=len(sequences), desc="Evaluating")
    for seq_idx, (seq_id, sequence) in enumerate(progress_bar):
        try:
            # Prepare MSA context if available
            has_context = False
            context_str = None
            if msa_dir:
                # Use sequence ID from FASTA record
                msa_path = Path(msa_dir) / f"{seq_id}.a3m"
                if msa_path.exists():
                    try:
                        context_str, _ = sample_context(
                            msa_path=str(msa_path),
                            max_num_samples=511,
                            max_token_length=14784,
                            max_query_similarity=0.95,
                            min_query_similarity=0.0,
                            neighbor_similarity_lower_bound=0.8,
                            seed=42,
                        )
                        has_context = True
                    except Exception as e:
                        logger.debug(
                            f"Error sampling context for sequence {seq_idx}: {e}"
                        )
                        context_str = None
                        has_context = False

            # Create masked sequence
            masked_seq = create_masked_sequence(sequence, mask_token, mlm_probability)

            # Prepare input strings for both original and masked sequences
            # We need to tokenize both to create proper labels
            if has_context and context_str is not None:
                original_input_str = context_str + "," + sequence
                masked_input_str = context_str + "," + masked_seq
            else:
                original_input_str = sequence
                masked_input_str = masked_seq

            # Prepare batches for both original and masked
            original_batch = batch_preparer.get_batch_kwargs(
                [original_input_str], device=device
            )
            masked_batch = batch_preparer.get_batch_kwargs(
                [masked_input_str], device=device
            )

            # Get input IDs and sequence IDs
            original_input_ids = original_batch["input_ids"][0]  # Shape: (seq_length,)
            masked_input_ids = masked_batch["input_ids"][0]  # Shape: (seq_length,)
            sequence_ids = masked_batch["sequence_ids"][0]  # Shape: (seq_length,)

            # Safety check: ensure both batches have the same length
            if original_input_ids.shape[0] != masked_input_ids.shape[0]:
                logger.warning(
                    f"Sequence {seq_idx}: Length mismatch between original and masked batches. "
                    f"Original: {original_input_ids.shape[0]}, Masked: {masked_input_ids.shape[0]}. Skipping."
                )
                continue

            # Find positions belonging to the last sequence (the query)
            last_seq_id = sequence_ids.max().item()
            query_mask = sequence_ids == last_seq_id

            # Get boundary token mask to exclude special tokens
            boundary_mask = batch_preparer.get_boundary_token_mask(
                masked_batch["input_ids"]
            )[0]
            valid_query_mask = query_mask & (~boundary_mask)

            # Create labels tensor (same shape as masked_input_ids)
            labels = torch.full_like(masked_input_ids, padding_idx)

            # Map original token IDs to masked positions
            # Get query sequence tokens from both original and masked
            original_query_tokens = original_input_ids[valid_query_mask]
            masked_query_tokens = masked_input_ids[valid_query_mask]

            # Find positions where tokens differ (these are masked positions)
            # Use pre-computed mask_token_id from batch_preparer
            masked_positions = (masked_query_tokens == mask_token_id) & (
                original_query_tokens != mask_token_id
            )

            # Set labels: original token IDs at masked positions, padding_idx elsewhere
            labels[valid_query_mask] = torch.where(
                masked_positions,
                original_query_tokens,
                torch.full_like(original_query_tokens, padding_idx),
            )

            # Run model forward pass with masked batch
            with torch.no_grad():
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    outputs = model(
                        input_ids=masked_batch["input_ids"],
                        within_seq_position_ids=masked_batch["within_seq_position_ids"],
                        global_position_ids=masked_batch["global_position_ids"],
                        sequence_ids=masked_batch["sequence_ids"],
                        labels=labels.unsqueeze(0),  # Add batch dimension
                        past_key_values=None,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                    )

            # Extract loss and logits
            loss = outputs.mlm_loss.item()
            logits = outputs.logits[0]  # Remove batch dimension

            # Compute accuracy on masked positions only
            # Get predictions for query sequence positions
            query_logits = logits[valid_query_mask]  # Shape: (query_len, vocab_size)
            query_labels = labels[valid_query_mask]  # Shape: (query_len,)

            # Only consider masked positions (where labels != padding_idx)
            masked_mask = query_labels != padding_idx
            if masked_mask.any():
                masked_logits = query_logits[masked_mask]
                masked_labels = query_labels[masked_mask]

                predicted_tokens = torch.argmax(masked_logits, dim=-1)
                correct = (predicted_tokens == masked_labels).sum().item()
                total = masked_labels.size(0)

                all_losses.append(loss)
                all_correct += correct
                all_total += total
            else:
                # No masked positions found - skip this sequence
                logger.debug(f"Sequence {seq_idx}: No masked positions found")
                continue

        except Exception as e:
            logger.warning(f"Error evaluating sequence {seq_idx}: {e}")
            continue

        # Update progress bar
        progress_bar.set_postfix(
            {
                "Processed": seq_idx + 1,
                "Loss": f"{np.mean(all_losses):.4f}" if all_losses else "N/A",
            }
        )

    # Compute overall metrics
    metrics = {}
    if len(all_losses) > 0:
        avg_loss = np.mean(all_losses)
        perplexity = np.exp(avg_loss)
        accuracy = all_correct / all_total if all_total > 0 else 0.0

        metrics["perplexity"] = perplexity
        metrics["accuracy"] = accuracy

    return metrics


def evaluate(config: Dict, output_path: Optional[str] = None):
    """
    Evaluate original E1 model on validation data.

    Args:
        config: Configuration dictionary
        output_path: Optional output path override
    """
    # Extract config
    general_conf = config["general"]
    data_conf = config["data"]
    train_conf = config["training"]

    set_seeds(general_conf["seed"])

    # Determine output directory
    if output_path is not None:
        output_dir = str(output_path)
    else:
        output_dir = train_conf.get("output_dir", "results/evaluation")
        output_dir = os.path.join(output_dir, "original_model_eval")

    # Create output directory early for logging
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Evaluating Original E1 Model (No Fine-tuning)")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")

    # Handle GPU selection
    gpu_id = general_conf.get("gpu_id", "0")
    if isinstance(gpu_id, int):
        gpu_id = str(gpu_id)
    elif isinstance(gpu_id, list):
        gpu_id = ",".join(map(str, gpu_id))

    if gpu_id.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        logger.info(f"Using GPU(s): {gpu_id}")
    else:
        num_gpus = torch.cuda.device_count()
        logger.info(f"Using all available GPUs: {num_gpus}")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data from FASTA file
    input_file = data_conf.get("input_file", None)
    if input_file is None:
        raise ValueError(
            "Config must specify 'data.input_file' field with path to FASTA file"
        )

    logger.info(f"Loading sequences from: {input_file}")
    sequences = []
    for record in SeqIO.parse(input_file, "fasta"):
        seq_str = str(record.seq)
        # Preprocess: Replace rare amino acids with X
        seq_str = (
            seq_str.replace("O", "X")
            .replace("B", "X")
            .replace("U", "X")
            .replace("Z", "X")
            .replace("J", "X")
        )
        sequences.append((record.id, seq_str))

    logger.info(f"Loaded {len(sequences)} sequences from FASTA file")

    # Get model dtype from config
    model_dtype = train_conf.get("model_dtype", None)

    # Get HuggingFace model identifier from config
    model_name = train_conf.get("checkpoint", None)
    if model_name is None:
        raise ValueError(
            "Config must specify 'checkpoint' field with HuggingFace model identifier "
            "(e.g., 'Profluent-Bio/E1-600m')"
        )

    # Load original model
    model, batch_preparer = load_original_model(
        model_name, device, model_dtype=model_dtype
    )

    # Get MSA directory from config (optional)
    msa_dir = data_conf.get("msa_dir", None)
    if msa_dir:
        logger.info(f"MSA directory: {msa_dir}")
    else:
        logger.info("No MSA directory specified - using single sequence mode")

    # Get max_eval_samples to limit evaluation (optional, for memory management)
    max_eval_samples = train_conf.get("max_eval_samples", None)
    if max_eval_samples is not None and len(sequences) > max_eval_samples:
        import random

        random.seed(general_conf["seed"])
        sequences = random.sample(sequences, max_eval_samples)
        logger.info(
            f"Limited evaluation set to {max_eval_samples} samples to prevent OOM"
        )

    # Get MLM probability
    mlm_probability = train_conf.get("mlm_probability", 0.15)

    logger.info("=" * 80)
    logger.info("Evaluation Configuration Summary")
    logger.info("=" * 80)
    logger.info(f"Model (HuggingFace): {model_name}")
    logger.info(f"Model dtype: {model_dtype if model_dtype else 'float32'}")
    logger.info(f"MLM probability: {mlm_probability}")
    logger.info(f"Input file: {input_file}")
    logger.info(
        f"MSA directory: {msa_dir if msa_dir else 'None (single sequence mode)'}"
    )
    logger.info(f"Total sequences to evaluate: {len(sequences)}")
    logger.info("=" * 80)
    logger.info("Starting evaluation...")

    # Evaluate all sequences
    eval_results = evaluate_sequence_batch(
        model=model,
        batch_preparer=batch_preparer,
        sequences=sequences,
        msa_dir=msa_dir,
        mlm_probability=mlm_probability,
        device=device,
    )

    # Add prefix to match expected format
    final_results = {}
    for key, value in eval_results.items():
        final_results[f"eval_{key}"] = value

    logger.info("=" * 80)
    logger.info("Evaluation Results")
    logger.info("=" * 80)
    for key, value in sorted(final_results.items()):
        logger.info(f"{key}: {value:.6f}")
    logger.info("=" * 80)

    # Save results to file
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("Original E1 Model Evaluation Results\n")
        f.write("=" * 80 + "\n")
        for key, value in sorted(final_results.items()):
            f.write(f"{key}: {value:.6f}\n")
        f.write("=" * 80 + "\n")
    logger.info(f"Results saved to {results_file}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    import time

    import yaml

    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Evaluate original E1 model (without fine-tuning)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Simple output path setup (similar to process_config but simplified)
    output_dir = config.get("training", {}).get("output_dir", "results/evaluation")
    output_path = os.path.join(output_dir, "original_model_eval")

    evaluate(config, output_path=output_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Total time used: {(time.time() - start_time)/60:.1f} minutes")
