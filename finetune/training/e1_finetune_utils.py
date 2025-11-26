import os
import numpy as np
import torch
import logging
from typing import Dict, Optional, Tuple

from peft import LoraConfig, get_peft_model
from modeling_e1 import E1ForMaskedLM, E1BatchPreparer
from training.finetune_utils import (
    _locate_offline_checkpoint,
    _resolve_hf_cache_dir,
    HF_CACHE_DIR,
)

logger = logging.getLogger(__name__)


def load_e1_model(
    checkpoint: str,
    lora_config: Dict,
    model_dtype: Optional[str] = None,
) -> Tuple[torch.nn.Module, E1BatchPreparer]:
    """
    Load E1 model and apply LoRA configuration.

    Args:
        checkpoint: HuggingFace model checkpoint path (e.g., "Synthyra/Profluent-E1-600M")
        lora_config: LoRA configuration dictionary
        model_dtype: Model dtype ("float16", "bfloat16", or None for float32)

    Returns:
        Tuple of (model, batch_preparer)
    """
    logger.info(f"Loading E1 model from {checkpoint}...")

    # Determine dtype
    dtype = None
    if model_dtype == "float16":
        dtype = torch.float16
    elif model_dtype == "bfloat16":
        dtype = torch.bfloat16

    # Resolve checkpoint path (local directory vs HF repo id) and cache location
    checkpoint_path = os.path.expanduser(checkpoint)
    checkpoint_is_local = os.path.isdir(checkpoint_path)

    if checkpoint_is_local:
        resolved_checkpoint = checkpoint_path
        cache_dir = None
    else:
        resolved_checkpoint = _locate_offline_checkpoint(checkpoint)
        cache_dir = (
            HF_CACHE_DIR if os.path.isdir(HF_CACHE_DIR) else _resolve_hf_cache_dir()
        )

        if resolved_checkpoint is None:
            resolved_checkpoint = checkpoint
            logger.warning(
                "Offline cache miss for '%s'. Attempting to load directly from HuggingFace ID.",
                checkpoint,
            )
        else:
            logger.info(
                "Resolved offline checkpoint for '%s' at '%s'",
                checkpoint,
                resolved_checkpoint,
            )

    if cache_dir:
        logger.info(f"Using HuggingFace cache directory: {cache_dir}")

    # Load model with trust_remote_code=True (required for E1 custom model classes)
    # Use local_files_only=True for offline environments
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if dtype is not None:
        load_kwargs["dtype"] = dtype

    try:
        model = E1ForMaskedLM.from_pretrained(resolved_checkpoint, **load_kwargs)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Create batch preparer (E1 models don't have tokenizer attribute like ESM)
    batch_preparer = E1BatchPreparer()

    # Log number of trainable parameters before LoRA
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"E1 Model - Trainable Parameters (before LoRA): {params/1e6:.1f}M")

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"E1 Model - Total Parameters: {total_params/1e6:.1f}M")

    # LoRA configuration
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

    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        bias=lora_config.get("bias", "none"),
        target_modules=default_target_modules,
        lora_dropout=lora_config.get("lora_dropout", 0.05),
    )

    # Log LoRA configuration
    logger.info("LoRA Configuration:")
    logger.info(f"  - rank (r): {peft_config.r}")
    logger.info(f"  - alpha: {peft_config.lora_alpha}")
    logger.info(f"  - dropout: {peft_config.lora_dropout}")
    logger.info(f"  - target_modules: {peft_config.target_modules}")
    logger.info(f"  - bias: {peft_config.bias}")

    # Apply PEFT LoRA
    model = get_peft_model(model, peft_config)

    # Ensure model is in training mode
    model.train()

    # Log trainable parameters after LoRA
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"E1 LoRA Model - Trainable Parameters: {trainable_params/1e6:.1f}M")
    logger.info(
        f"E1 LoRA Model - Trainable Percentage: {100*trainable_params/total_params:.2f}%"
    )

    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found! Check LoRA configuration.")

    # Verify that LoRA adapters are properly set up
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    return model, batch_preparer
