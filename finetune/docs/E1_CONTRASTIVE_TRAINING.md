# E1 Contrastive Binding Pipeline

## Overview

This document describes the contrastive fine-tuning pipeline for residue-level metal ion binding. The pipeline combines:
- **Prototype-aligned contrastive learning**: class prototypes anchor the embedding space.
- **Binary classification** using prototype distance scores.
- **Optional MLM regularization** to preserve the pretrained language signal.
- **LoRA** for parameter-efficient backbone adaptation plus K-fold CV for evaluation.

## Training Goals
- Encourage **variant-invariant residue embeddings** through multi-view contrastive loss.
- **Align embeddings to fixed class prototypes**; negative prototype is the negation of the positive prototype.
- Train **residue-level logits** from prototype distance scores for calibrated probabilities.
- Regularize with **MLM** to reduce drift from the pretrained manifold.

## Architecture & Data Flow

### Model Stack
```
E1ForContrastiveBinding (finetune/training/e1_contrastive_model.py)
├── _original_model (LoRA-wrapped E1ForMaskedLM)
├── Frozen prototypes (per ion): pos_prototype, neg_prototype = -pos_prototype
└── Contrastive head uses hidden states directly (no extra projector)
```

### Batch & Views
1) **Collation** (`e1_contrastive_collator.py`):
   - Creates `num_variants` masked views per sequence with random mask prob in `[mask_prob_min, mask_prob_max]`.
   - Optionally resamples an independent MSA context per view (`contrastive.resample_msa_per_variant`); contexts are padded to the longest view in the batch so shapes stay `[batch, n_views, seq_len]`.
   - Returns per-view `mlm_labels`, `binding_labels`, `label_mask`, and `contrastive_label_mask` (excludes masked tokens from contrastive/BCE).
2) **Backbone forward**:
   - Loops over views and runs a separate backbone forward for each masked input (no sequence concatenation); averages MLM loss across views.
   - Hidden states are stacked to `[batch, n_views, seq_len, hidden]`.
3) **Loss inputs**:
   - Valid residues from `label_mask` are extracted; each residue keeps all views for contrastive + prototype losses.
   - MLM runs on the flattened hidden states.

### Prototype Strategy
- Positive prototype is initialized from **mean positive residue embeddings** (first `max_samples` batches, single unmasked view).
- Stored as unit vectors; negative prototype is its negation (symmetry).
- Prototypes are **frozen after init** (EMA knobs are ignored).

## Losses

- **Unsupervised contrastive (NT-Xent style)** on normalized residue embeddings across views (masked positions are excluded via `contrastive_label_mask`).
  - Weight: `contrastive.unsupervised_weight`
  - Temperature: `contrastive.temperature`
- **Prototype alignment** (log-softmax over two prototypes, applied only when margin violated):
  - Margins: `contrastive.eps_pos`, `contrastive.eps_neg`
  - Weight: `contrastive.prototype_weight`
  - Diagnostics: `pull_mask_ratio`, average similarities to pos/neg prototypes.
- **BCE on prototype distance scores**:
  - Scores = `(sim_pos - sim_neg) / scoring_temperature`, averaged across views.
  - Weight: `contrastive.bce_weight`
  - Optional label smoothing: `contrastive.label_smoothing`
- **MLM**:
  - Uses pretrained `mlm_head`; computed independently per view and averaged; weight `training.mlm_weight`.

Total loss = weighted sum of the above.

## Training Loop (per ion)
1) Build datasets and dataloaders (train/val) with contrastive collator.
2) **Prototype init**: compute positive prototype from early positive residues; set on the model.
3) For each epoch:
   - Train with gradient accumulation; log component losses + prototype diagnostics.
   - Validate using prototype distance scores; compute AUPRC/HRA/F1/MCC.
   - Track best epoch by HRA and save checkpoints (LoRA adapters + heads).
4) Repeat for all ions; cross-validation handled by `run_cv`.

## Configuration Highlights (`finetune/configs/e1_contrastive.yaml`)
- `training`:
  - `ions`: ion list; CV folds; LR, weight decay, warmup, batch/accum.
  - `mlm_weight`: set >0 to enable MLM regularization.
  - `compile_flex_attention`: optional torch.compile for attention kernels.
- `contrastive`:
  - `num_variants`, `mask_prob_min/max`: view generation.
  - `prototype_dim`: defaults to hidden size.
  - `prototype_weight`, `unsupervised_weight`, `bce_weight`: loss weights.
  - `temperature`: contrastive temperature.
  - `eps_pos`, `eps_neg`: prototype margins.
  - `scoring_temperature`: scales logits used for BCE/metrics.
  - `label_smoothing`: applied to BCE targets during training.
- `lora` / `model`:
  - LoRA rank/alpha/targets; dropout; checkpoint name; dtype.

## Running Training
Single GPU:
```bash
source .venv/bin/activate
python finetune/train_e1_contrastive.py --config finetune/configs/e1_contrastive.yaml
```

DDP (example 4 GPUs):
```bash
torchrun --nproc_per_node=4 finetune/train_e1_contrastive.py \
    --config finetune/configs/e1_contrastive.yaml
```

Outputs are written under `training.output_dir/{timestamp}/fold_{k}` with adapters, heads, plots, and logs.

## Key Implementation Notes
- Prototypes are **unit-norm**; embeddings are normalized inside the loss before similarity.
- Prototype diagnostics (`sim_pos`, `sim_neg`) are **averaged over all residues** (both positive and negative); a negative `sim_pos` often just reflects class imbalance, not prototype drift.
- Contrastive loss returns zero if only one view is present (e.g., validation).
- BCE logits use the **difference** between pos/neg prototype similarities, promoting calibrated scores.
- Label masks keep loss on query residues; masked tokens for MLM do not remove positions from prototype/BCE losses.

## Caveats & Tips
- Prototype init uses **early batches only**; if positives are rare, consider increasing `max_samples` in `initialize_prototypes`.
- Because prototypes are frozen, large LoRA updates can rotate embeddings; monitor `pull_mask_ratio` to ensure alignment pressure is active.
- `sim_pos` trending negative is expected once negatives dominate; focus on AUPRC/HRA and `pull_mask_ratio` for health.
- If OOM, reduce `num_variants`, `max_total_tokens`, or increase `accum_steps`.
- DDP requires `ddp_find_unused_parameters=true` because only one ion head is active per forward.

## Output Structure
```
results/e1_contrastive/{run_id}/
├── fold_1/
│   ├── best_e1_contrastive_model_lora/    # PEFT adapters
│   ├── best_e1_contrastive_model_heads.pt # heads + metadata
│   └── plots/                             # loss/PR curves
├── ... fold_k
├── oof_preds.npz
├── global_thresholds.yaml
└── training.log
```

## Troubleshooting Checklist
- `sim_pos` negative but metrics stable → likely class imbalance; confirm `pull_mask_ratio` > 0 for some steps.
- Low AUPRC + pull_ratio ≈ 0 → increase `prototype_weight` or margins (`eps_pos/eps_neg`) to enforce alignment.
- NaN/Inf warnings → check for invalid prototypes; initialization logs should confirm a non-empty positive set.
