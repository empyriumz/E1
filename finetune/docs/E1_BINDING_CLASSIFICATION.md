# E1 Binding Classification Fine-tuning Pipeline

## Overview

This document describes the E1 LoRA-based fine-tuning pipeline for residue-level metal ion binding site classification. The pipeline now trains jointly with:

1. **LoRA** on the E1 backbone for parameter-efficient updates
2. **Ion-specific classification heads** for residue-level BCE
3. **Auxiliary MLM loss** using the pretrained `mlm_head` from `E1ForMaskedLM` (never re-initialized)
4. **Joint collator** that masks tokens for MLM and excludes those masked tokens from the BCE loss
5. **DDP** for multi-GPU and **K-fold CV** for robust evaluation

## Architecture

### Model Structure (Joint BCE + MLM)

```
E1ForJointBindingMLM (inherits E1ForResidueClassification -> E1ForMaskedLM)
├── _original_model (PeftModel / E1ForMaskedLM with LoRA)
│   ├── mlm_head (pretrained, reused)
│   └── model (E1Model with LoRA-injected layers)
│       └── layers[0..N] with LoRA on q/k/v/o and w1/w2/w3
├── e1_backbone (E1ForMaskedLM reference)
└── classifier_heads (ModuleDict of ion-specific heads)
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| Main Training Script | `train_e1_binding.py` | K-fold CV orchestration; joint loss when `training.mlm_weight > 0` |
| Joint Model | `training/e1_joint_model.py` | Reuses pretrained `mlm_head`; combines BCE + MLM |
| Classification Model | `training/e1_classification_model.py` | Inherits `E1ForMaskedLM`; requires `mlm_head` to exist |
| Trainer | `training/e1_binding_trainer.py` | Logs BCE/MLM components; single forward with masked inputs |
| Dataset | `training/e1_binding_dataset.py` | Data loading with MSA context |
| Joint Collator | `training/e1_joint_collator.py` | Masks query tokens for MLM and removes those tokens from BCE |
| LoRA Utils | `training/e1_finetune_utils.py` | LoRA application to E1 |

## How LoRA Works

### Initialization

When `get_peft_model()` is called, PEFT injects LoRA adapters into the target modules:

1. Target linear layers (e.g., `q_proj`) are replaced with `LoraLayer` wrappers
2. Each `LoraLayer` contains:
   - `base_layer`: Original frozen weights
   - `lora_A`: Low-rank down-projection (initialized with small random values)
   - `lora_B`: Low-rank up-projection (**initialized to ZERO**)

### Forward Pass

```python
# LoraLayer.forward():
output = base_layer(x) + lora_B(lora_A(x)) * (alpha / r)
```

**Important**: Since `lora_B` starts at zero, the model initially behaves identically to the base model. During training:
1. `lora_B` receives gradients first (directly from the loss)
2. Once `lora_B` becomes non-zero, `lora_A` also starts receiving gradients
3. Both matrices are updated throughout training

### Trainable Parameters

With default configuration:
- LoRA rank (r): 16
- Target modules: q_proj, k_proj, v_proj, o_proj, w1, w2, w3
- Trainable parameters: ~2.5M (0.4% of total)
- Plus classification heads: ~10K per ion type

## Usage (Joint Training)

### Single GPU Training

```bash
cd /host/E1
source .venv/bin/activate
python finetune/train_e1_binding.py --config finetune/configs/e1_binding_config.yaml
```

### Multi-GPU Training (DDP)

```bash
# Using 4 GPUs (joint BCE+MLM)
torchrun --nproc_per_node=4 finetune/train_e1_binding.py \
    --config finetune/configs/e1_binding_config.yaml
```

DDP automatically:
- Splits data across GPUs using `DistributedSampler`
- Synchronizes gradients during backward pass
- Handles model parameter updates

### Configuration (joint knobs)

Key options in `e1_binding_config.yaml` (or `e1_binding.yaml`):

```yaml
training:
  ions: ["CA", "ZN", "MG"]  # Ion types to train
  epochs: 30
  batch_size: 2
  accum_steps: 4  # Effective batch = 8
  learning_rate: 0.0001
  num_folds: 5
  ddp_find_unused_parameters: true  # For multi-head models
  mlm_weight: 0.4        # >0 enables joint training
  mlm_probability: 0.15  # Masking rate on query tokens

lora:
  r: 16                    # LoRA rank
  alpha: 32                # Scaling factor
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "w1"
    - "w2"
    - "w3"
```

### Data & Collation Notes

- Binding labels align to query tokens; context/special tokens use ignore_index.
- Joint collator masks query tokens for MLM and **excludes those masked positions from BCE** (clears `label_mask`, sets `binding_labels` to ignore_index there).
- Unlabeled sequences (labels omitted/None) are allowed and become MLM-only rows.

## Data Format

### FASTA File Format

```
>protein_id
MKFLILLFNILCLFPVLAADNHGVGPQG
0100110000000000100000000000
```

- Line 1: Header with protein ID
- Line 2: Amino acid sequence
- Line 3: Binary labels (0=non-binding, 1=binding)

### MSA Context (Optional)

MSA files in A3M format provide homolog sequences for context. During training:
- Context sequences are sampled stochastically
- Sampling varies per epoch (data augmentation)
- Max context is controlled by `max_token_length` and `max_num_samples`

### REE Dataset Construction

The REE (Rare Earth Elements) ion type combines data from LREE and HREE:
- **Union of protein IDs**: REE contains all proteins from both LREE and HREE
- **Label combination**: For proteins in both datasets, labels are combined with logical OR
- **Shared train/val splits**: To prevent data leakage, LREE, HREE, and REE use the same K-fold partition

> [!IMPORTANT]
> The pipeline ensures that if a protein is in the validation fold for REE, it is also in the validation fold for LREE and HREE. This prevents a protein from being used for training in LREE/HREE while being held out for REE validation.

## Testing

Run the test suite to verify the pipeline:

```bash
# Full test suite
python finetune/tests/test_e1_binding_pipeline.py

# Quick LoRA verification
python finetune/tests/test_lora_corrected.py
```

### Test Output Interpretation

```
=== Test: Modify BOTH lora_A and lora_B ===
Modified lora_A mean: 1.000000
Modified lora_B mean: 1.000000
Hidden state difference: 1176.0
✓ LoRA IS working!
```

If the hidden state difference is 0, check:
1. `lora_B` initialization (should be zero initially)
2. That you're modifying BOTH lora_A and lora_B for testing
3. PEFT installation and version

## Common Issues

1) **Masked tokens hurting BCE:** Use the joint collator; it already removes masked tokens from BCE targets.

2) **Need for pretrained MLM head:** Models inherit `E1ForMaskedLM` and require `mlm_head` to exist (no random init). Load an `E1ForMaskedLM` checkpoint with LoRA applied.

3) **OOM errors:** Reduce `batch_size` or `max_token_length`, increase `accum_steps`, or enable gradient checkpointing.

4) **DDP hanging:** Keep `ddp_find_unused_parameters: true` since only one ion head is active per forward. Early stopping is now synchronized across ranks automatically.

## Performance Considerations

### Memory Usage

| Configuration | GPU Memory (approx) |
|--------------|---------------------|
| batch_size=1, max_tokens=4096 | ~8GB |
| batch_size=2, max_tokens=8192 | ~16GB |
| batch_size=4, max_tokens=8192 | ~28GB |

### Training Speed

| Setup | Samples/sec (approx) |
|-------|---------------------|
| 1x A100 80GB | ~5 |
| 4x A100 80GB (DDP) | ~18 |

## Output Structure

```
results/e1_binding/
├── fold_1/
│   ├── best_e1_binding_model_lora/   # PEFT LoRA adapters (~2-3MB)
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   ├── best_e1_binding_model_heads.pt  # Classifier heads + metadata (~100KB)
│   └── plots/
│       ├── loss_fold_1_CA.png
│       └── auprc_fold_1_CA.png
├── fold_2/
│   └── ...
├── oof_preds.npz           # Aggregated OOF predictions
├── global_thresholds.yaml  # Global thresholds per ion computed from OOF
├── config.yaml             # Copy of training configuration
└── training.log
```

### Out-of-Fold (OOF) predictions

- During K-fold CV, the best epoch per fold (by validation HRA) snapshots OOF predictions.
- After all folds complete, aggregated OOF predictions are saved to `{output_dir}/oof_preds.npz`.
- Contents:
  - `ids`: per-residue identifiers formatted as `protein_id:position` (0-based position within the query sequence).
  - `labels`: ground-truth binary labels (float array).
  - `probs`: predicted probabilities from the best checkpoint of each fold (float array).
  - `folds`: fold index for each record (int array).
  - `ions`: ion type for each record (object array).
- Global thresholds are automatically computed from OOF predictions and saved to `global_thresholds.yaml`.
- Validation uses `label_smoothing=0` by config, so labels in OOF are unsmoothed.

### Checkpoint Contents

The checkpoint is split into two parts for storage efficiency:

**LoRA Adapter Directory** (`*_lora/`):
- `adapter_config.json`: LoRA configuration
- `adapter_model.safetensors`: LoRA weights (~2-3MB)

**Heads Checkpoint** (`*_heads.pt`):
```python
{
    "epoch": 15,
    "base_model_checkpoint": "Synthyra/Profluent-E1-600M",
    "lora_config": {"r": 16, "alpha": 32, ...},
    "classifier_heads": {...},  # Ion-specific head weights
    "ions": ["CA", "ZN", "MG"],
    "metrics": {"auprc": 0.85, "hra": 0.82},
    "fold": 1,
}
```

### Loading Checkpoints for Inference

```python
from training.e1_checkpoint_utils import load_lora_checkpoint, load_ensemble_models

# Load single fold
model = load_lora_checkpoint(
    "results/e1_binding/fold_1/best_e1_binding_model",
    device=torch.device("cuda"),
)

# Load all 5 folds for ensemble inference
models = load_ensemble_models(
    "results/e1_binding/",
    num_folds=5,
    device=torch.device("cuda"),
)
```

## Test Set Inference

After training completes, run inference on a held-out test set using the ensemble of K models:

### Basic Usage

```bash
# Using explicit test FASTA path with {ION} placeholder
python finetune/infer_e1_binding.py \
    --run_dir results/e1_binding/2025-12-09-13-11 \
    --test_fasta /path/to/test_data/{ION}_test.fasta \
    --output_dir results/test_evaluation

# With custom MSA directory
python finetune/infer_e1_binding.py \
    --run_dir results/e1_binding/2025-12-09-13-11 \
    --test_fasta /path/to/test_data/{ION}_test.fasta \
    --msa_dir /path/to/test_msa/{ION}
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--run_dir` | Path to training run directory | Required |
| `--output_dir` | Directory for results | `{run_dir}/test_results` |
| `--test_fasta` | Test FASTA path (supports `{ION}` placeholder) | From config |
| `--msa_dir` | Test MSA directory (supports `{ION}` placeholder) | From config |
| `--device` | CUDA device | `cuda:0` |
| `--num_folds` | Number of folds to load | From config |
| `--batch_size` | Inference batch size | `4` |

### Expected Files in `run_dir`

The inference script expects the following files:
- `config.yaml`: Training configuration
- `global_thresholds.yaml`: Thresholds computed from OOF predictions
- `fold_*/best_e1_binding_model_lora/`: LoRA adapters per fold
- `fold_*/best_e1_binding_model_heads.pt`: Classifier heads per fold

### Output Structure

```
{output_dir}/
├── test_metrics.csv         # Per-ion metrics
├── summary.yaml             # Overall summary with averages
├── test_predictions.npz     # Raw predictions per ion
├── inference.log            # Inference log
└── plots/
    └── threshold_analysis_*.png  # ROC and PR curves per ion
```

### Metrics Computed

- **Threshold-independent**: AUC-ROC, AUPRC, High-recall AUPRC (R≥0.7, R≥0.8)
- **Threshold-dependent** (using global threshold): F1, MCC, Precision, Recall, Confusion Matrix

## API Reference

### E1ForResidueClassification

```python
class E1ForResidueClassification(nn.Module):
    """
    E1 model with ion-specific classification heads.
    
    Args:
        e1_model: PeftModel wrapping E1ForMaskedLM with LoRA
        ion_types: List of ion types (e.g., ["CA", "ZN"])
        dropout: Dropout rate for classification heads
        freeze_backbone: Whether to freeze the backbone (not needed with LoRA)
    
    Forward Args:
        input_ids: Token IDs (batch_size, seq_len)
        within_seq_position_ids: Position within each sequence
        global_position_ids: Global positions
        sequence_ids: Sequence identifiers
        ion: Ion type to classify (e.g., "CA")
        labels: Binary labels (optional)
        label_mask: Valid position mask (optional)
        pos_weight: Positive class weight for BCE loss (optional)

    
    Returns:
        E1ResidueClassificationOutput with loss, logits, last_hidden_state
    """
```

### E1BindingTrainer

```python
class E1BindingTrainer:
    """
    Trainer for E1 binding classification with DDP support.
    
    Args:
        model: E1ForResidueClassification (or DDP-wrapped)
        conf: Configuration object
        device: Training device
        is_distributed: Whether using DDP
        world_size: Number of processes
        rank: Current process rank
        pos_weights: Dictionary of positive class weights per ion
    
    Key Methods:
        create_dataloaders(): Create train/val loaders with DistributedSampler
        train_epoch(): Run one training epoch
        validate_epoch(): Run validation
        save_checkpoint(): Save model (handles DDP unwrapping)
    """
```

## Version History

- **v1.0**: Initial implementation with LoRA and K-fold CV
- **v1.1**: Added DDP support for multi-GPU training
- **v1.2**: Added REE ion support (LREE + HREE combined)
- **v1.3**: Fixed NCCL timeout by synchronizing early stopping across ranks
- **v1.4**: Fixed data leakage by ensuring LREE, HREE, and REE use shared train/val splits
- **v1.5**: LoRA-only checkpointing (~99% storage reduction); configurable `train_mlm_head` option
- **v1.6**: Refactored `pos_weight` to be stateless and passed dynamically during forward pass

