import sys
import os
import torch
import torch.nn as nn
from typing import List, Dict, Any

# Add project root to path
project_root = "/sdcc/u/xdai/E1"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "finetune"))

# Mocking necessary imports if they fail or require complex setup
from modeling_e1 import E1Config, E1ForMaskedLM, E1BatchPreparer
from training.e1_contrastive_model import E1ForContrastiveBinding
from train_e1_contrastive import (
    load_e1_for_contrastive,
)  # Try to reuse if possible, or mock
from peft import LoraConfig, get_peft_model


def get_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def profile_run(
    enable_checkpointing: bool,
    n_views: int = 4,
    seq_len: int = 512,
    batch_size: int = 1,
):
    print(
        f"\n--- Profiling with Checkpointing={enable_checkpointing}, Views={n_views} ---"
    )

    # Clean up
    torch.cuda.empty_cache()
    reset_peak_memory()

    # 1. Setup Model
    # Using a smaller config for speed/memory if we are on CPU, but we prefer CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Config similar to real one
    config = E1Config(
        hidden_size=1024,  # Reduced for test speed/OOM avoidance on small debug GPUs
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=33,  # Standard E1 vocab
        max_num_positions_within_seq=8192,
        # gradient_checkpointing=enable_checkpointing # We set this via enable_gradient_checkpointing()
    )

    # Create base model
    base_model = E1ForMaskedLM(config)

    # Apply LoRA (mimic training)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["mlm_head"],
    )
    base_model = get_peft_model(base_model, peft_config)

    # Wrap in Contrastive Model
    model = E1ForContrastiveBinding(e1_model=base_model, ion_types=["ZN"], dropout=0.1)

    model.to(device)
    model.train()

    # --- VERIFICATION START ---
    print("\n--- Parameter Verification ---")
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"Total Parameters: {len(trainable_params) + len(frozen_params)}")
    print(f"Trainable Parameters: {len(trainable_params)}")
    print(f"Frozen Parameters: {len(frozen_params)}")

    print("Sample Trainable Params:")
    for name in trainable_params[:5]:
        print(f" - {name}")

    print("Sample Frozen Params:")
    for name in frozen_params[:5]:
        print(f" - {name}")

    if len(trainable_params) > 1000:  # Heuristic
        print("WARNING: Too many trainable parameters! Check LoRA freezing.")

    # Check if backbone is frozen
    backbone_frozen = all(
        "backbone" in name or "model.layers" in name
        for name in frozen_params
        if "lora" not in name and "head" not in name
    )  # Rough check
    # Actually, just check if layers.0.self_attn.q_proj.weight is frozen (it should be replaced by lora, but the base weight should be frozen)
    # Lora model structure: model.model.layers...

    print("--- End Verification ---\n")
    # --- VERIFICATION END ---

    # 2. Setup Dummy Data
    # Shape: [batch, n_views, seq_len]
    input_ids = torch.randint(low=2, high=30, size=(batch_size, n_views, seq_len)).to(
        device
    )
    # Correct shape for other inputs
    # E1ForContrastiveBinding receives standard E1 inputs but with [batch, n_views, seq_len] shape?
    # No, let's check forward signature.
    # forward(input_ids, within_seq_position_ids, ...) where input_ids is [batch, seq_len] usually.
    # BUT the Collator produces [batch, num_variants, seq_len].
    # Wait, the collator output for input_ids is [batch, num_variants, seq_len].
    # But E1ForContrastiveBinding.forward expects inputs that match the collator output?
    # Let's check E1ForContrastiveBinding.forward again.

    # E1ForContrastiveBinding.forward(self, input_ids, ...)
    # If input_ids is 3D [batch, n_views, seq_len], it must handle it.
    # Check lines viewed earlier:
    # "for v in range(n_views): ... backbone_kwargs = {k: v[:, v_idx, ...] ...}"
    # Yes, it iterates.

    within_seq_position_ids = (
        torch.arange(seq_len)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, n_views, seq_len)
        .to(device)
    )
    global_position_ids = within_seq_position_ids.clone()
    sequence_ids = torch.zeros_like(within_seq_position_ids)

    # Labels
    # Binding labels: [batch, seq_len] typically, expanded?
    # The forward method signature implies standard HF style.
    # But usually models take [batch, seq_len].
    # If the collator produces (batch, views, seq), the model must handle it.

    # 3. Forward Pass
    print("Running Forward...")
    reset_peak_memory()
    initial_mem = get_peak_memory()

    # Prepare inputs
    inputs = {
        "input_ids": input_ids,
        "within_seq_position_ids": within_seq_position_ids,
        "global_position_ids": global_position_ids,
        "sequence_ids": sequence_ids,
        "ion": "ZN",
        # Labels are optional for pure forward, but we want to simulate loss calculation
        # "labels": torch.randint(0, 2, (batch_size, seq_len)).float().to(device) # Labels are typically shared across views?
    }

    try:
        outputs = model(**inputs)
        loss = outputs.loss
        print(f"Forward successful. Loss: {loss.item()}")

        mem_after_forward = get_peak_memory()
        print(f"Peak Memory after Forward: {mem_after_forward:.2f} MB")

        # 4. Backward Pass
        print("Running Backward...")
        loss.backward()
        mem_after_backward = get_peak_memory()
        print(f"Peak Memory after Backward: {mem_after_backward:.2f} MB")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available. Profiling on CPU (not representative of OOM)."
        )

    try:
        profile_run(enable_checkpointing=False)
        profile_run(enable_checkpointing=True)
    except Exception as e:
        print(e)
