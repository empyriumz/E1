"""
Data collator for E1 residue-level binding classification.

This module provides E1DataCollatorForResidueClassification, which handles:
- Converting protein sequences + MSA context to E1 batch format
- Aligning per-residue binding labels with E1 token positions
- Properly handling special tokens and context sequence masking
"""

import torch
from typing import List, Dict, Any, Optional
import logging

from modeling_e1 import E1BatchPreparer

logger = logging.getLogger(__name__)


class E1DataCollatorForResidueClassification:
    """
    Data collator for E1 residue-level binding classification.

    Handles:
    - Converting comma-separated sequence strings to E1 batch format
    - Aligning per-residue binary labels with E1 token positions
    - Setting ignore_index for context sequences and special tokens
    - Truncating sequences that exceed max_total_tokens to prevent OOM

    E1 Tokenization Example:
        Sequence: "ABC" (3 residues)
        Tokens: <bos> 1 A B C 2 <eos>
        Token IDs: [0, 1, 2, 3, 4, 5, 6] (indices)

        Labels: [0, 1, 0] (per residue)
        Aligned Labels: [ignore, ignore, 0, 1, 0, ignore, ignore]
                        ^bos    ^1        ^A ^B ^C ^2       ^eos
    """

    # Constants for special token positions
    SPECIAL_TOKENS_PREFIX = 2  # <bos> and 1
    SPECIAL_TOKENS_SUFFIX = 2  # 2 and <eos>

    def __init__(
        self,
        max_total_tokens: int = 8192,
        max_query_tokens: int = 1024,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
    ):
        """
        Initialize E1DataCollatorForResidueClassification.

        Args:
            max_total_tokens: Maximum total tokens per sample including context + query.
                              Samples exceeding this will have context sequences removed.
            max_query_tokens: Maximum tokens for query sequence.
                              Query sequences exceeding this will be truncated.
            ignore_index: Label value for positions to ignore during loss computation.
            label_smoothing: Amount of label smoothing to apply (0.0 = no smoothing).
                             Labels become: 0 -> smoothing, 1 -> 1-smoothing.
        """
        self.batch_preparer = E1BatchPreparer()
        self.pad_token_id = self.batch_preparer.pad_token_id
        self.max_total_tokens = max_total_tokens
        self.max_query_tokens = max_query_tokens
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

        # Get boundary token IDs for special token detection
        self.boundary_token_ids = self.batch_preparer.boundary_token_ids

        logger.info(f"E1DataCollatorForResidueClassification initialized:")
        logger.info(f"  - Pad token ID: {self.pad_token_id}")
        logger.info(f"  - Ignore index: {ignore_index}")
        logger.info(f"  - Max total tokens: {max_total_tokens}")
        logger.info(f"  - Max query tokens: {max_query_tokens}")
        if label_smoothing > 0:
            logger.info(f"  - Label smoothing: {label_smoothing}")

    def _truncate_sequence_and_labels(
        self, sequence: str, labels: List[int], msa_context: Optional[str] = None
    ) -> tuple:
        """
        Truncate sequence and labels to fit within token limits.

        Args:
            sequence: Query protein sequence (amino acid string)
            labels: Per-residue binary labels (same length as sequence)
            msa_context: Optional comma-separated context sequences

        Returns:
            Tuple of (truncated_full_sequence, truncated_labels)
        """
        # Each sequence gets 4 special tokens: <bos>, 1, ..., 2, <eos>
        tokens_per_seq = 4

        # Truncate query sequence if needed
        max_query_chars = self.max_query_tokens - tokens_per_seq
        if len(sequence) > max_query_chars:
            sequence = sequence[:max_query_chars]
            labels = labels[:max_query_chars]

        # Calculate total tokens
        query_tokens = len(sequence) + tokens_per_seq

        # Build full sequence with context
        if msa_context and msa_context.strip():
            context_seqs = msa_context.split(",")
            context_tokens = sum(len(seq) + tokens_per_seq for seq in context_seqs)
            total_tokens = context_tokens + query_tokens

            # Remove context sequences from beginning if needed
            while context_seqs and total_tokens > self.max_total_tokens:
                removed = context_seqs.pop(0)
                total_tokens -= len(removed) + tokens_per_seq

            if context_seqs:
                full_sequence = ",".join(context_seqs) + "," + sequence
            else:
                full_sequence = sequence
        else:
            full_sequence = sequence

        return full_sequence, labels

    def _align_labels_to_tokens(
        self,
        input_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
        labels_list: List[List[int]],
    ) -> torch.Tensor:
        """
        Align per-residue labels to E1 token positions.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            sequence_ids: Sequence IDs (batch_size, seq_len) - identifies which sequence each token belongs to
            labels_list: List of per-residue labels for each sample

        Returns:
            aligned_labels: (batch_size, seq_len) with ignore_index for non-residue positions
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize all labels as ignore_index
        aligned_labels = torch.full(
            (batch_size, seq_len),
            self.ignore_index,
            dtype=torch.float,
            device=device,
        )

        # Get boundary token IDs on correct device
        boundary_token_ids = self.boundary_token_ids.to(device)

        for batch_idx in range(batch_size):
            # Find the query sequence (last sequence in multi-sequence input)
            # Query sequence has the maximum sequence_id
            sample_seq_ids = sequence_ids[batch_idx]
            max_seq_id = sample_seq_ids.max().item()

            # Get mask for query sequence tokens
            query_mask = sample_seq_ids == max_seq_id

            # Get mask for boundary tokens (special tokens)
            boundary_mask = torch.isin(input_ids[batch_idx], boundary_token_ids)

            # Valid residue positions: query sequence AND not boundary token
            valid_residue_mask = query_mask & (~boundary_mask)

            # Get indices of valid residue positions
            valid_indices = torch.where(valid_residue_mask)[0]

            # Get the labels for this sample
            sample_labels = labels_list[batch_idx]

            # Align labels to valid positions
            # The number of valid positions should match the number of labels
            num_labels = len(sample_labels)
            num_valid = len(valid_indices)

            if num_labels != num_valid:
                logger.warning(
                    f"Batch {batch_idx}: Label count ({num_labels}) != valid positions ({num_valid}). "
                    f"Truncating/padding as needed."
                )
                # Handle mismatch by taking minimum
                num_to_assign = min(num_labels, num_valid)
            else:
                num_to_assign = num_labels

            # Assign labels to valid positions (with optional label smoothing)
            for i in range(num_to_assign):
                label_val = float(sample_labels[i])
                # Apply label smoothing: 0 -> smoothing, 1 -> 1-smoothing
                if self.label_smoothing > 0:
                    if label_val == 1.0:
                        label_val = 1.0 - self.label_smoothing
                    else:
                        label_val = self.label_smoothing
                aligned_labels[batch_idx, valid_indices[i]] = label_val

        return aligned_labels

    def _create_label_mask(self, aligned_labels: torch.Tensor) -> torch.Tensor:
        """
        Create boolean mask for valid label positions.

        Args:
            aligned_labels: Aligned labels tensor with ignore_index for invalid positions

        Returns:
            label_mask: Boolean tensor, True for positions with valid labels
        """
        return aligned_labels != self.ignore_index

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            examples: List of dictionaries, each containing:
                - "sequence": Protein sequence string
                - "labels": List of per-residue binary labels
                - "msa_context": Optional comma-separated context sequences

        Returns:
            Dictionary with batched tensors ready for E1 model forward pass:
                - input_ids: Token IDs
                - sequence_ids: Sequence identifier for each token
                - within_seq_position_ids: Position within each sequence
                - global_position_ids: Global position IDs
                - binding_labels: Aligned binding labels (float tensor)
                - label_mask: Boolean mask for valid label positions
        """
        # Process each example
        sequences = []
        labels_list = []

        for ex in examples:
            sequence = ex["sequence"]
            labels = ex["labels"]
            msa_context = ex.get("msa_context", None)

            # Truncate if needed
            full_seq, truncated_labels = self._truncate_sequence_and_labels(
                sequence, labels, msa_context
            )

            sequences.append(full_seq)
            labels_list.append(truncated_labels)

        # Get E1 batch tensors using batch preparer
        batch_dict = self.batch_preparer.get_batch_kwargs(
            sequences, device=torch.device("cpu")
        )

        # Extract tensors
        input_ids = batch_dict["input_ids"]
        sequence_ids = batch_dict["sequence_ids"]
        within_seq_position_ids = batch_dict["within_seq_position_ids"]
        global_position_ids = batch_dict["global_position_ids"]

        # Align labels to token positions
        aligned_labels = self._align_labels_to_tokens(
            input_ids, sequence_ids, labels_list
        )

        # Create label mask
        label_mask = self._create_label_mask(aligned_labels)

        return {
            "input_ids": input_ids,
            "sequence_ids": sequence_ids,
            "within_seq_position_ids": within_seq_position_ids,
            "global_position_ids": global_position_ids,
            "binding_labels": aligned_labels,
            "label_mask": label_mask,
        }


class E1DataCollatorForResidueClassificationWithPosWeight(
    E1DataCollatorForResidueClassification
):
    """
    Extended data collator that also computes pos_weight for BCE loss.

    This is useful when you want to compute pos_weight per batch dynamically,
    though typically pos_weight is computed from the full training set.
    """

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch and compute pos_weight."""
        batch = super().__call__(examples)

        # Compute pos_weight from batch labels
        valid_labels = batch["binding_labels"][batch["label_mask"]]
        if len(valid_labels) > 0:
            pos_count = valid_labels.sum().item()
            neg_count = len(valid_labels) - pos_count
            if pos_count > 0:
                pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float)
            else:
                pos_weight = torch.tensor([1.0], dtype=torch.float)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float)

        batch["pos_weight"] = pos_weight

        return batch
