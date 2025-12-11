"""
Joint data collator for binding (BCE) + MLM auxiliary loss.

Builds on the existing binding collator to keep label alignment intact, then
applies the same masking strategy used by E1DataCollatorForMLM to produce
`mlm_labels` and masked `input_ids`. Only query-sequence, non-boundary tokens
are eligible for masking.
"""

from typing import Any, Dict, List

import torch
from modeling_e1 import E1BatchPreparer
from training.e1_binding_collator import E1DataCollatorForResidueClassification


class E1DataCollatorForJointBindingMLM:
    """
    Collate batches for joint binding (BCE) + MLM training.

    Outputs:
        - input_ids: masked input ids for MLM
        - mlm_labels: original token ids at masked positions, pad token elsewhere
        - sequence_ids / position ids (unchanged)
        - binding_labels: aligned binding labels (ignore_index for non-query/boundary)
        - label_mask: valid positions for BCE
    """

    def __init__(
        self,
        mlm_probability: float = 0.15,
        max_total_tokens: int = 8192,
        max_query_tokens: int = 1024,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        self.binding_collator = E1DataCollatorForResidueClassification(
            max_total_tokens=max_total_tokens,
            max_query_tokens=max_query_tokens,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        # Reuse preparer from binding collator for token ids / vocab info
        self.batch_preparer: E1BatchPreparer = self.binding_collator.batch_preparer
        self.mlm_probability = mlm_probability
        self.mask_token_id = self.batch_preparer.mask_token_id
        self.pad_token_id = self.batch_preparer.pad_token_id
        self.vocab_size = len(self.batch_preparer.vocab)
        self.boundary_token_ids = self.batch_preparer.boundary_token_ids
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def _apply_mlm_masking(
        self, input_ids: torch.Tensor, sequence_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply MLM masking on query tokens only (exclude boundary tokens).
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        boundary_token_ids = self.boundary_token_ids.to(device)
        boundary_mask = torch.isin(input_ids, boundary_token_ids)

        # Query sequence = max sequence_id per sample
        query_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        for i in range(batch_size):
            max_seq_id = sequence_ids[i].max().item()
            query_mask[i] = sequence_ids[i] == max_seq_id

        valid_mask = query_mask & (~boundary_mask)

        masking_probs = torch.rand(batch_size, seq_len, device=device)
        selected_mask = (masking_probs < self.mlm_probability) & valid_mask

        mask_decision = torch.rand(batch_size, seq_len, device=device)
        mask_token_mask = (mask_decision < 0.8) & selected_mask
        random_token_mask = (
            (mask_decision >= 0.8) & (mask_decision < 0.9) & selected_mask
        )
        # 10% unchanged: no special mask needed beyond selected_mask

        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_token_mask] = self.mask_token_id

        random_tokens = torch.randint(
            0,
            self.vocab_size,
            (batch_size, seq_len),
            device=device,
            dtype=input_ids.dtype,
        )
        masked_input_ids[random_token_mask] = random_tokens[random_token_mask]

        mlm_labels = torch.full_like(input_ids, self.pad_token_id)
        mlm_labels[selected_mask] = input_ids[selected_mask]

        return {
            "input_ids": masked_input_ids,
            "mlm_labels": mlm_labels,
            "selected_mask": selected_mask,
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch for joint training.

        Supports unlabeled examples by allowing missing/None "labels" entries.
        """
        # Fill missing labels with all-ignore to allow MLM-only rows
        processed_examples: List[Dict[str, Any]] = []
        for ex in examples:
            seq = ex["sequence"]
            labels = ex.get("labels")
            if labels is None:
                labels = [-100] * len(seq)  # ignore everywhere
            processed_examples.append({**ex, "labels": labels})

        batch = self.binding_collator(processed_examples)

        # Apply MLM masking on top of binding-collated tensors
        mlm = self._apply_mlm_masking(
            input_ids=batch["input_ids"], sequence_ids=batch["sequence_ids"]
        )

        # Avoid training BCE on masked positions
        label_mask = batch["label_mask"]
        selected_mask = mlm["selected_mask"].to(label_mask.device)
        label_mask = label_mask & (~selected_mask)
        binding_labels = batch["binding_labels"]
        binding_labels = binding_labels.masked_fill(
            selected_mask, float(self.ignore_index)
        )

        batch["input_ids"] = mlm["input_ids"]
        batch["mlm_labels"] = mlm["mlm_labels"]
        batch["label_mask"] = label_mask
        batch["binding_labels"] = binding_labels

        return batch
