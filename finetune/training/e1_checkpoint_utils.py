"""
Utilities for loading LoRA-only checkpoints for inference.

This module provides functions to reconstruct the full model from:
1. Base E1 model checkpoint (from HuggingFace or local)
2. LoRA adapter directory (saved by PEFT's save_pretrained)
3. Classifier heads checkpoint (saved as _heads.pt)
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging

from peft import PeftModel
from modeling_e1 import E1ForMaskedLM, E1BatchPreparer
from training.e1_classification_model import E1ForResidueClassification
from training.e1_joint_model import E1ForJointBindingMLM
from training.finetune_utils import _locate_offline_checkpoint, _resolve_hf_cache_dir

logger = logging.getLogger(__name__)


def load_base_e1_model(
    checkpoint: str,
    model_dtype: Optional[str] = None,
) -> E1ForMaskedLM:
    """
    Load base E1 model without LoRA adapters.

    Args:
        checkpoint: HuggingFace model checkpoint or local path
        model_dtype: Optional dtype ("float16", "bfloat16", None for float32)

    Returns:
        E1ForMaskedLM model
    """
    logger.info(f"Loading base E1 model from {checkpoint}...")

    # Determine dtype
    dtype = None
    if model_dtype == "float16":
        dtype = torch.float16
    elif model_dtype == "bfloat16":
        dtype = torch.bfloat16

    # Resolve checkpoint path
    checkpoint_path = os.path.expanduser(checkpoint)
    if os.path.isdir(checkpoint_path):
        resolved_checkpoint = checkpoint_path
        cache_dir = None
    else:
        resolved_checkpoint = _locate_offline_checkpoint(checkpoint)
        cache_dir = _resolve_hf_cache_dir()
        if resolved_checkpoint is None:
            resolved_checkpoint = checkpoint

    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if dtype is not None:
        load_kwargs["dtype"] = dtype

    model = E1ForMaskedLM.from_pretrained(resolved_checkpoint, **load_kwargs)
    return model


def load_lora_checkpoint(
    checkpoint_base_path: str,
    base_model_checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
    model_dtype: Optional[str] = "bfloat16",
    mlm_weight: float = 0.0,
) -> Union[E1ForResidueClassification, E1ForJointBindingMLM]:
    """
    Load a complete model from LoRA-only checkpoint for inference.

    This reconstructs the full model from:
    1. Base E1 model
    2. LoRA adapters (from {checkpoint_base_path}_lora/)
    3. Classifier heads (from {checkpoint_base_path}_heads.pt)

    Args:
        checkpoint_base_path: Base path to checkpoint (e.g., "fold_1/best_e1_binding_model")
        base_model_checkpoint: Base model checkpoint. If None, reads from heads checkpoint.
        device: Device to load model to
        model_dtype: Model dtype for base model
        mlm_weight: If > 0, returns E1ForJointBindingMLM, else E1ForResidueClassification

    Returns:
        Reconstructed model ready for inference
    """
    # Clean up path
    base_path = str(checkpoint_base_path).replace(".pt", "").replace("_heads", "")
    lora_dir = f"{base_path}_lora"
    heads_path = f"{base_path}_heads.pt"

    # Load heads checkpoint first to get metadata
    if not os.path.exists(heads_path):
        raise FileNotFoundError(f"Heads checkpoint not found: {heads_path}")

    heads_ckpt = torch.load(heads_path, map_location="cpu")

    # Get base model checkpoint
    if base_model_checkpoint is None:
        base_model_checkpoint = heads_ckpt.get("base_model_checkpoint")
        if base_model_checkpoint is None:
            raise ValueError(
                "base_model_checkpoint not provided and not found in checkpoint. "
                "Please specify the base model checkpoint path."
            )

    # Get ion types
    ion_types = heads_ckpt.get("ions", ["CA", "ZN", "MG"])

    logger.info(f"Loading model from LoRA checkpoint:")
    logger.info(f"  - Base model: {base_model_checkpoint}")
    logger.info(f"  - LoRA adapters: {lora_dir}")
    logger.info(f"  - Ions: {ion_types}")

    # Load base model
    base_model = load_base_e1_model(base_model_checkpoint, model_dtype=model_dtype)

    # Load LoRA adapters
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(f"LoRA adapter directory not found: {lora_dir}")

    peft_model = PeftModel.from_pretrained(base_model, lora_dir)
    logger.info(f"LoRA adapters loaded from {lora_dir}")

    # Create classification model
    if mlm_weight > 0:
        model = E1ForJointBindingMLM(
            e1_model=peft_model,
            ion_types=ion_types,
            dropout=0.1,
            mlm_weight=mlm_weight,
            freeze_backbone=False,
        )
    else:
        model = E1ForResidueClassification(
            e1_model=peft_model,
            ion_types=ion_types,
            dropout=0.1,
            freeze_backbone=False,
        )

    # Load classifier heads
    if "classifier_heads" in heads_ckpt:
        model.classifier_heads.load_state_dict(heads_ckpt["classifier_heads"])
        logger.info("Classifier heads loaded")

    # Move to device
    if device is not None:
        dtype = None
        if model_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif model_dtype == "float16":
            dtype = torch.float16
        model = model.to(device=device, dtype=dtype)

    # Set to eval mode
    model.eval()

    logger.info("Model reconstruction complete")
    return model


def load_ensemble_models(
    checkpoint_dir: str,
    num_folds: int = 5,
    base_model_checkpoint: Optional[str] = None,
    device: Optional[torch.device] = None,
    model_dtype: str = "bfloat16",
    mlm_weight: float = 0.0,
) -> List[Union[E1ForResidueClassification, E1ForJointBindingMLM]]:
    """
    Load all fold models for ensemble inference.

    Args:
        checkpoint_dir: Directory containing fold_1/, fold_2/, etc.
        num_folds: Number of folds to load
        base_model_checkpoint: Base model checkpoint (if None, reads from checkpoints)
        device: Device to load models to
        model_dtype: Model dtype
        mlm_weight: MLM weight (>0 for joint model)

    Returns:
        List of loaded models, one per fold
    """
    models = []

    for fold_idx in range(1, num_folds + 1):
        fold_path = Path(checkpoint_dir) / f"fold_{fold_idx}" / "best_e1_binding_model"

        if not fold_path.parent.exists():
            logger.warning(f"Fold {fold_idx} directory not found, skipping")
            continue

        try:
            model = load_lora_checkpoint(
                checkpoint_base_path=str(fold_path),
                base_model_checkpoint=base_model_checkpoint,
                device=device,
                model_dtype=model_dtype,
                mlm_weight=mlm_weight,
            )
            models.append(model)
            logger.info(f"Loaded fold {fold_idx} model")
        except Exception as e:
            logger.error(f"Failed to load fold {fold_idx}: {e}")
            raise

    logger.info(f"Loaded {len(models)} fold models for ensemble")
    return models


def get_checkpoint_info(checkpoint_base_path: str) -> Dict[str, Any]:
    """
    Get information about a LoRA checkpoint without loading the model.

    Args:
        checkpoint_base_path: Base path to checkpoint

    Returns:
        Dictionary with checkpoint metadata
    """
    base_path = str(checkpoint_base_path).replace(".pt", "").replace("_heads", "")
    heads_path = f"{base_path}_heads.pt"
    lora_dir = f"{base_path}_lora"

    info = {
        "heads_path": heads_path,
        "lora_dir": lora_dir,
        "heads_exists": os.path.exists(heads_path),
        "lora_exists": os.path.isdir(lora_dir),
    }

    if info["heads_exists"]:
        heads_ckpt = torch.load(heads_path, map_location="cpu")
        info["epoch"] = heads_ckpt.get("epoch")
        info["ions"] = heads_ckpt.get("ions")
        info["metrics"] = heads_ckpt.get("metrics")
        info["base_model_checkpoint"] = heads_ckpt.get("base_model_checkpoint")

    if info["lora_exists"]:
        adapter_config = Path(lora_dir) / "adapter_config.json"
        info["adapter_config_exists"] = adapter_config.exists()

        # Get adapter file size
        adapter_files = list(Path(lora_dir).glob("*"))
        info["lora_files"] = [f.name for f in adapter_files]
        info["lora_size_mb"] = sum(f.stat().st_size for f in adapter_files) / (
            1024 * 1024
        )

    return info
