"""
Data collator for contrastive learning with multiple masked variants.

This collator generates num_variants masked "views" of each sequence for
unsupervised contrastive learning. Each view has independent random masking.
"""

from typing import Any, Dict, List, Optional

from E1.msa_sampling import sample_context

import torch
from modeling_e1 import E1BatchPreparer
from training.e1_binding_collator import E1DataCollatorForResidueClassification


class E1DataCollatorForContrastive:
    """
    Collate batches with multiple masked variants for contrastive learning.

    For each sample:
    1. Tokenize sequence with MSA context (using base collator)
    2. Generate num_variants copies with different random masks
    3. Apply masking probability sampled uniformly from [mask_prob_min, mask_prob_max]

    Output tensors have shape [batch, num_variants, seq_len] for per-view data.
    Binding labels are shared across views: [batch, seq_len].
    """

    def __init__(
        self,
        num_variants: int = 4,
        mask_prob_min: float = 0.05,
        mask_prob_max: float = 0.15,
        max_total_tokens: int = 8192,
        max_query_tokens: int = 1024,
        ignore_index: int = -100,
        resample_msa_per_variant: bool = True,
        msa_sampling_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize contrastive collator.

        Args:
            num_variants: Number of masked views per sample
            mask_prob_min: Minimum masking probability
            mask_prob_max: Maximum masking probability
            max_total_tokens: Maximum total tokens
            max_query_tokens: Maximum query sequence tokens
            ignore_index: Label index to ignore in loss computation
        """
        self.num_variants = num_variants
        self.mask_prob_min = mask_prob_min
        self.mask_prob_max = mask_prob_max
        self.ignore_index = ignore_index
        self.resample_msa_per_variant = resample_msa_per_variant
        self.msa_sampling_config = msa_sampling_config or {}

        # Use base binding collator for tokenization (no label smoothing - handled in loss)
        self.binding_collator = E1DataCollatorForResidueClassification(
            max_total_tokens=max_total_tokens,
            max_query_tokens=max_query_tokens,
            ignore_index=ignore_index,
        )
        self.batch_preparer: E1BatchPreparer = self.binding_collator.batch_preparer

        # Token IDs for masking
        self.mask_token_id = self.batch_preparer.mask_token_id
        self.pad_token_id = self.batch_preparer.pad_token_id
        self.vocab_size = len(self.batch_preparer.vocab)
        self.boundary_token_ids = self.batch_preparer.boundary_token_ids

    def _apply_random_masking(
        self, input_ids: torch.Tensor, sequence_ids: torch.Tensor, mask_prob: float
    ) -> Dict[str, torch.Tensor]:
        """
        Apply random MLM masking to query tokens.

        Args:
            input_ids: [batch, seq_len] token IDs
            sequence_ids: [batch, seq_len] sequence identifiers
            mask_prob: Probability of masking each token

        Returns:
            Dict with masked_input_ids and mlm_labels
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Identify boundary tokens (not eligible for masking)
        boundary_token_ids = self.boundary_token_ids.to(device)
        boundary_mask = torch.isin(input_ids, boundary_token_ids)

        # Query sequence = max sequence_id per sample
        query_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        for i in range(batch_size):
            max_seq_id = sequence_ids[i].max().item()
            query_mask[i] = sequence_ids[i] == max_seq_id

        # Valid positions for masking: query tokens that are not boundaries
        valid_mask = query_mask & (~boundary_mask)

        # Sample positions to mask
        masking_probs = torch.rand(batch_size, seq_len, device=device)
        selected_mask = (masking_probs < mask_prob) & valid_mask

        # Apply 80-10-10 masking strategy
        mask_decision = torch.rand(batch_size, seq_len, device=device)
        mask_token_mask = (mask_decision < 0.8) & selected_mask
        random_token_mask = (
            (mask_decision >= 0.8) & (mask_decision < 0.9) & selected_mask
        )
        # 10% unchanged

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

        # MLM labels: original tokens at masked positions, pad elsewhere
        mlm_labels = torch.full_like(input_ids, self.pad_token_id)
        mlm_labels[selected_mask] = input_ids[selected_mask]

        return {
            "input_ids": masked_input_ids,
            "mlm_labels": mlm_labels,
            "mask_positions": selected_mask,
        }

    def _maybe_resample_msa(
        self, example: Dict[str, Any], variant_idx: int
    ) -> Optional[str]:
        """
        Optionally resample MSA context per variant if msa_path is provided.
        """
        if not self.resample_msa_per_variant:
            return example.get("msa_context")

        msa_path = example.get("msa_path")
        if not msa_path:
            return example.get("msa_context")

        try:
            seed = self.msa_sampling_config.get("seed_offset", 0) + variant_idx
            context_str, _ = sample_context(
                msa_path=msa_path,
                max_num_samples=self.msa_sampling_config.get("max_num_samples", 32),
                max_token_length=self.msa_sampling_config.get("max_token_length", 8192),
                max_query_similarity=self.msa_sampling_config.get(
                    "max_query_similarity", 0.95
                ),
                min_query_similarity=self.msa_sampling_config.get(
                    "min_query_similarity", 0.0
                ),
                neighbor_similarity_lower_bound=self.msa_sampling_config.get(
                    "neighbor_similarity_lower_bound", 0.8
                ),
                seed=seed,
                device=torch.device("cpu"),
            )
            return context_str
        except Exception:
            # Fallback to existing context if sampling fails
            return example.get("msa_context")

    def _pad_to_length(
        self, tensor: torch.Tensor, target_len: int, pad_value: int
    ) -> torch.Tensor:
        """
        Pad a [batch, seq_len] tensor to target_len on the sequence dimension.
        """
        if tensor.size(1) == target_len:
            return tensor
        pad_amount = target_len - tensor.size(1)
        return torch.nn.functional.pad(tensor, (0, pad_amount), value=pad_value)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch with multiple masked variants.

        Args:
            examples: List of samples with sequence, labels, msa_context, protein_id

        Returns:
            Dict with tensors:
                - input_ids: [batch, num_variants, seq_len]
                - within_seq_position_ids: [batch, num_variants, seq_len]
                - global_position_ids: [batch, num_variants, seq_len]
                - sequence_ids: [batch, num_variants, seq_len]
                - mlm_labels: [batch, num_variants, seq_len]
                - binding_labels: [batch, seq_len] (shared across views)
                - label_mask: [batch, seq_len] (shared across views)
                - contrastive_label_mask: [batch, num_variants, seq_len] (excludes masked positions)
                - protein_ids: List[str]
        """
        if self.resample_msa_per_variant:
            variant_batches = []
            for v in range(self.num_variants):
                variant_examples: List[Dict[str, Any]] = []
                for ex in examples:
                    variant_examples.append(
                        {
                            "sequence": ex["sequence"],
                            "labels": ex["labels"],
                            "msa_context": self._maybe_resample_msa(ex, v),
                            "protein_id": ex.get("protein_id"),
                        }
                    )
                variant_batches.append(self.binding_collator(variant_examples))

            # Pad all variant batches to common length
            max_seq_len = max(b["input_ids"].size(1) for b in variant_batches)
            padded_batches = []
            for b in variant_batches:
                padded_batches.append(
                    {
                        "input_ids": self._pad_to_length(
                            b["input_ids"], max_seq_len, self.pad_token_id
                        ),
                        "sequence_ids": self._pad_to_length(
                            b["sequence_ids"], max_seq_len, 0
                        ),
                        "within_seq_position_ids": self._pad_to_length(
                            b["within_seq_position_ids"], max_seq_len, 0
                        ),
                        "global_position_ids": self._pad_to_length(
                            b["global_position_ids"], max_seq_len, 0
                        ),
                        "binding_labels": self._pad_to_length(
                            b["binding_labels"], max_seq_len, self.ignore_index
                        ),
                        "label_mask": self._pad_to_length(
                            b["label_mask"], max_seq_len, 0
                        ).bool(),
                        "protein_ids": b.get("protein_ids", []),
                    }
                )

            all_input_ids = []
            all_mlm_labels = []
            all_sequence_ids = []
            all_within_pos = []
            all_global_pos = []
            all_binding_labels = []
            all_label_masks = []
            all_mask_positions = []

            for v in range(self.num_variants):
                batch_v = padded_batches[v]
                mask_prob = (
                    torch.empty(1)
                    .uniform_(self.mask_prob_min, self.mask_prob_max)
                    .item()
                )
                masked_result = self._apply_random_masking(
                    input_ids=batch_v["input_ids"],
                    sequence_ids=batch_v["sequence_ids"],
                    mask_prob=mask_prob,
                )

                all_input_ids.append(masked_result["input_ids"])
                all_mlm_labels.append(masked_result["mlm_labels"])
                all_mask_positions.append(masked_result["mask_positions"])
                all_sequence_ids.append(batch_v["sequence_ids"])
                all_within_pos.append(batch_v["within_seq_position_ids"])
                all_global_pos.append(batch_v["global_position_ids"])
                all_binding_labels.append(batch_v["binding_labels"])
                all_label_masks.append(batch_v["label_mask"])

            input_ids = torch.stack(all_input_ids, dim=1)  # [batch, n_views, seq_len]
            mlm_labels = torch.stack(all_mlm_labels, dim=1)
            sequence_ids = torch.stack(all_sequence_ids, dim=1)
            within_seq_position_ids = torch.stack(all_within_pos, dim=1)
            global_position_ids = torch.stack(all_global_pos, dim=1)
            binding_labels = torch.stack(all_binding_labels, dim=1)
            label_mask = torch.stack(all_label_masks, dim=1)
            mask_positions = torch.stack(all_mask_positions, dim=1)
        else:
            # First, get base collated batch (single view)
            base_batch = self.binding_collator(examples)

            # Initialize lists for multi-variant tensors
            all_input_ids = []
            all_mlm_labels = []
            all_mask_positions = []

            # Generate num_variants masked copies
            for _ in range(self.num_variants):
                # Sample masking probability for this variant
                mask_prob = (
                    torch.empty(1)
                    .uniform_(self.mask_prob_min, self.mask_prob_max)
                    .item()
                )

                masked_result = self._apply_random_masking(
                    input_ids=base_batch["input_ids"],
                    sequence_ids=base_batch["sequence_ids"],
                    mask_prob=mask_prob,
                )

                all_input_ids.append(masked_result["input_ids"])
                all_mlm_labels.append(masked_result["mlm_labels"])
                all_mask_positions.append(masked_result["mask_positions"])

            # Stack variants: [batch, num_variants, seq_len]
            input_ids = torch.stack(all_input_ids, dim=1)
            mlm_labels = torch.stack(all_mlm_labels, dim=1)
            mask_positions = torch.stack(all_mask_positions, dim=1)

            # Expand position/sequence IDs to match: [batch, num_variants, seq_len]
            within_seq_position_ids = (
                base_batch["within_seq_position_ids"]
                .unsqueeze(1)
                .expand(-1, self.num_variants, -1)
            )
            global_position_ids = (
                base_batch["global_position_ids"]
                .unsqueeze(1)
                .expand(-1, self.num_variants, -1)
            )
            sequence_ids = (
                base_batch["sequence_ids"]
                .unsqueeze(1)
                .expand(-1, self.num_variants, -1)
            )
            binding_labels = (
                base_batch["binding_labels"].unsqueeze(1).expand_as(input_ids)
            )
            label_mask = base_batch["label_mask"].unsqueeze(1).expand_as(input_ids)

        contrastive_label_mask = label_mask & (~mask_positions)

        protein_ids = None
        if self.resample_msa_per_variant:
            protein_ids = variant_batches[0].get("protein_ids", [])
        else:
            protein_ids = base_batch.get("protein_ids", [])

        return {
            "input_ids": input_ids,
            "within_seq_position_ids": within_seq_position_ids.contiguous(),
            "global_position_ids": global_position_ids.contiguous(),
            "sequence_ids": sequence_ids.contiguous(),
            "mlm_labels": mlm_labels,
            "binding_labels": binding_labels,
            "label_mask": label_mask,
            "contrastive_label_mask": contrastive_label_mask,
            "protein_ids": protein_ids,
        }
