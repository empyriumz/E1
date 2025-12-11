import os
import sys

# Add src directory to Python path so E1 module can be imported without installing the package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import logging
from typing import Dict, Optional

import torch

# Import reusable functions from evaluate_original_model_hf
from evaluate_original_model_hf import evaluate_sequence_batch, set_seeds, setup_logging
from modeling_e1 import E1ForMaskedLM
from peft import LoraConfig, PeftModel, get_peft_model

# Import utilities for checkpoint resolution
from training.finetune_utils import (
    HF_CACHE_DIR,
    _locate_offline_checkpoint,
    _resolve_hf_cache_dir,
)

# Module-level logger (will be configured in main function)
logger = logging.getLogger(__name__)


def load_finetuned_model(
    base_model_name: str,
    adapter_checkpoint: str,
    lora_config: Optional[Dict] = None,
    device: torch.device = None,
    model_dtype: Optional[str] = None,
):
    """
    Load finetuned E1 model with LoRA adapter.

    Args:
        base_model_name: HuggingFace model identifier (e.g., "Synthyra/Profluent-E1-600M")
        adapter_checkpoint: Path to directory containing adapter weights (e.g., "results/e1_lora_checkpoints/2025-11-25-21-19/best_checkpoint")
        lora_config: LoRA configuration dictionary. If None, will try to load from adapter_checkpoint/adapter_config.json
        device: Device to load model on (if None, will use CUDA if available)
        model_dtype: Model dtype ("float16", "bfloat16", or None for float32)

    Returns:
        Tuple of (model, batch_preparer)
    """
    from modeling_e1 import E1BatchPreparer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading base E1 model from HuggingFace: {base_model_name}...")
    logger.info(f"Loading LoRA adapter from: {adapter_checkpoint}...")

    # Determine dtype
    dtype = None
    if model_dtype == "float16":
        dtype = torch.float16
    elif model_dtype == "bfloat16":
        dtype = torch.bfloat16

    # Resolve base model checkpoint path
    checkpoint_path = os.path.expanduser(base_model_name)
    checkpoint_is_local = os.path.isdir(checkpoint_path)

    if checkpoint_is_local:
        resolved_checkpoint = checkpoint_path
        cache_dir = None
    else:
        resolved_checkpoint = _locate_offline_checkpoint(base_model_name)
        cache_dir = (
            HF_CACHE_DIR if os.path.isdir(HF_CACHE_DIR) else _resolve_hf_cache_dir()
        )

        if resolved_checkpoint is None:
            resolved_checkpoint = base_model_name
            logger.warning(
                "Offline cache miss for '%s'. Attempting to load directly from HuggingFace ID.",
                base_model_name,
            )
        else:
            logger.info(
                "Resolved offline checkpoint for '%s' at '%s'",
                base_model_name,
                resolved_checkpoint,
            )

    if cache_dir:
        logger.info(f"Using HuggingFace cache directory: {cache_dir}")

    # Load base model with trust_remote_code=True for custom model classes
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if dtype is not None:
        load_kwargs["dtype"] = dtype

    try:
        base_model = E1ForMaskedLM.from_pretrained(resolved_checkpoint, **load_kwargs)
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        raise

    # Log base model parameters
    total_params = sum(p.numel() for p in base_model.parameters())
    logger.info(f"Base Model - Total Parameters: {total_params/1e6:.1f}M")

    # Expand adapter checkpoint path
    adapter_path = os.path.expanduser(adapter_checkpoint)
    if not os.path.isdir(adapter_path):
        raise ValueError(
            f"Adapter checkpoint must be a directory: {adapter_path}. "
            f"Expected to contain 'adapter_model.safetensors' or 'adapter_model.bin'"
        )

    # Check if adapter files exist
    adapter_model_file = os.path.join(adapter_path, "adapter_model.safetensors")
    adapter_config_file = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_model_file) and not os.path.exists(
        os.path.join(adapter_path, "adapter_model.bin")
    ):
        raise ValueError(
            f"Adapter model file not found in {adapter_path}. "
            f"Expected 'adapter_model.safetensors' or 'adapter_model.bin'"
        )

    # Load LoRA configuration
    if lora_config is None:
        # Try to load from adapter_config.json
        if os.path.exists(adapter_config_file):
            import json

            logger.info(f"Loading LoRA config from {adapter_config_file}")
            with open(adapter_config_file, "r") as f:
                adapter_config_data = json.load(f)
            # Extract LoRA config from adapter config
            lora_config = {
                "r": adapter_config_data.get("r", 16),
                "alpha": adapter_config_data.get("lora_alpha", 32),
                "bias": adapter_config_data.get("bias", "none"),
                "lora_dropout": adapter_config_data.get("lora_dropout", 0.05),
                "target_modules": adapter_config_data.get("target_modules", None),
            }
            logger.info("Loaded LoRA config from adapter checkpoint")
        else:
            # Use default LoRA config if not provided and not found
            logger.warning(
                f"LoRA config not found at {adapter_config_file}. Using default config."
            )
            lora_config = {}

    # Default target_modules for E1 (can be overridden in config)
    default_target_modules = lora_config.get("target_modules", None)
    if default_target_modules is None:
        # E1 default: attention + MLP layers
        default_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "w1",
            "w2",
            "w3",
        ]

    # Load adapter using PeftModel.from_pretrained
    # This will automatically load the adapter config and weights from the checkpoint
    logger.info(f"Loading LoRA adapter from {adapter_path}...")

    # Try to load adapter directly (preferred method)
    # PeftModel.from_pretrained loads both the adapter config and weights
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)

        # Log LoRA configuration from loaded adapter
        if hasattr(model, "peft_config") and model.peft_config:
            peft_config = list(model.peft_config.values())[0]
            logger.info("LoRA Configuration (loaded from adapter):")
            logger.info(f"  - rank (r): {peft_config.r}")
            logger.info(f"  - alpha: {peft_config.lora_alpha}")
            logger.info(f"  - dropout: {peft_config.lora_dropout}")
            logger.info(f"  - target_modules: {peft_config.target_modules}")
            logger.info(f"  - bias: {peft_config.bias}")
    except Exception as e:
        # If loading fails (e.g., missing adapter_config.json), try alternative approach
        logger.warning(f"Direct adapter loading failed: {e}")
        logger.info("Attempting alternative loading method with provided config...")

        # Create LoRA config for PEFT
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            bias=lora_config.get("bias", "none"),
            target_modules=default_target_modules,
            lora_dropout=lora_config.get("lora_dropout", 0.05),
        )

        # Log LoRA configuration
        logger.info("LoRA Configuration (from provided config):")
        logger.info(f"  - rank (r): {peft_config.r}")
        logger.info(f"  - alpha: {peft_config.lora_alpha}")
        logger.info(f"  - dropout: {peft_config.lora_dropout}")
        logger.info(f"  - target_modules: {peft_config.target_modules}")
        logger.info(f"  - bias: {peft_config.bias}")

        # Apply LoRA to base model
        model = get_peft_model(base_model, peft_config)

        # Load trained adapter weights
        logger.info(f"Loading adapter weights from {adapter_path}...")
        try:
            model.load_adapter(adapter_path)
        except Exception as e2:
            logger.error(f"Failed to load adapter weights: {e2}")
            raise

    # Move model to device
    model = model.to(device)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Finetuned Model - Trainable Parameters: {trainable_params/1e6:.1f}M")
    logger.info(
        f"Finetuned Model - Trainable Percentage: {100*trainable_params/total_params:.2f}%"
    )

    # Create batch preparer
    batch_preparer = E1BatchPreparer()

    # Set model to evaluation mode
    model.eval()

    return model, batch_preparer


def evaluate(config: Dict, output_path: Optional[str] = None):
    """
    Evaluate finetuned E1 model (with LoRA adapter) on validation data.

    Args:
        config: Configuration dictionary
        output_path: Optional output path override
    """
    from Bio import SeqIO
    from modeling_e1 import compile_flex_attention_if_enabled

    # Extract config
    general_conf = config["general"]
    data_conf = config["data"]
    train_conf = config["training"]
    lora_conf = config.get("lora", {})

    set_seeds(general_conf["seed"])

    # Determine output directory
    if output_path is not None:
        output_dir = str(output_path)
    else:
        output_dir = train_conf.get("output_dir", "results/evaluation")

    # Create output directory early for logging
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Evaluating Finetuned E1 Model (with LoRA Adapter)")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")

    # Handle GPU selection
    gpu_id = general_conf.get("gpu_id", "0")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
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

    # Get base model identifier from config
    base_model_name = train_conf.get("checkpoint", None)
    if base_model_name is None:
        raise ValueError(
            "Config must specify 'training.checkpoint' field with HuggingFace model identifier "
            "(e.g., 'Profluent-Bio/E1-600m' or 'Synthyra/Profluent-E1-600M')"
        )

    # Get adapter checkpoint path
    adapter_checkpoint = train_conf.get("adapter_checkpoint", None)
    if adapter_checkpoint is None:
        raise ValueError(
            "Config must specify 'training.adapter_checkpoint' field with path to LoRA adapter directory "
            "(e.g., 'results/e1_lora_checkpoints/2025-11-25-21-19/best_checkpoint')"
        )

    # Load finetuned model with LoRA adapter
    model, batch_preparer = load_finetuned_model(
        base_model_name=base_model_name,
        adapter_checkpoint=adapter_checkpoint,
        lora_config=lora_conf if lora_conf else None,
        device=device,
        model_dtype=model_dtype,
    )

    # Compile flex_attention if enabled (for inference only)
    compile_flex_attention = train_conf.get("compile_flex_attention", False)
    if compile_flex_attention:
        logger.info("Attempting to compile flex_attention for inference...")
        compile_flex_attention_if_enabled(enabled=True)
    else:
        logger.info("Skipping flex_attention compilation (default behavior)")

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
    logger.info(f"Base Model (HuggingFace): {base_model_name}")
    logger.info(f"Adapter Checkpoint: {adapter_checkpoint}")
    logger.info(f"Model dtype: {model_dtype if model_dtype else 'float32'}")
    logger.info(f"Compile flex_attention: {compile_flex_attention}")
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
        f.write("Finetuned E1 Model Evaluation Results (with LoRA Adapter)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Base Model: {base_model_name}\n")
        f.write(f"Adapter Checkpoint: {adapter_checkpoint}\n")
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
        description="Evaluate finetuned E1 model (with LoRA adapter)"
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

    # Simple output path setup
    output_path = config.get("training", {}).get("output_dir", "results/evaluation")

    evaluate(config, output_path=output_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Total time used: {(time.time() - start_time)/60:.1f} minutes")
