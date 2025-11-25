import torch
from typing import List, Dict
import logging

from modeling_e1 import E1BatchPreparer

logger = logging.getLogger(__name__)


class E1DataCollatorForMLM:
    """
    Data collator for E1 Masked Language Modeling.

    Handles:
    - Converting comma-separated sequence strings to E1 batch format
    - Applying BERT-style masking (15% of tokens, with 80/10/10 split)
    - Only masking the query sequence (last sequence), not context sequences
    - Excluding boundary tokens from masking
    - Truncating sequences that exceed max_total_tokens to prevent OOM
    """

    def __init__(
        self,
        mlm_probability: float = 0.15,
        mask_token: str = "?",
        max_total_tokens: int = 8192,
        max_query_tokens: int = 1024,
    ):
        """
        Initialize E1DataCollatorForMLM.

        Args:
            mlm_probability: Probability of masking each token (default 0.15)
            mask_token: Token to use for masking (default "?" for E1)
            max_total_tokens: Maximum total tokens per sample including context + query.
                              Samples exceeding this will have context sequences removed
                              to fit within the limit. (default 16384)
            max_query_tokens: Maximum tokens for query sequence (default 4096).
                              This is critical because the query uses full O(n²) self-attention
                              while context uses efficient block-causal attention.
                              Query sequences exceeding this will be truncated.
        """
        self.batch_preparer = E1BatchPreparer()
        self.mlm_probability = mlm_probability
        self.mask_token = mask_token
        self.mask_token_id = self.batch_preparer.mask_token_id
        self.pad_token_id = self.batch_preparer.pad_token_id
        self.max_total_tokens = max_total_tokens
        self.max_query_tokens = max_query_tokens

        # Get vocabulary size for random token replacement
        self.vocab_size = len(self.batch_preparer.vocab)

        logger.info(f"E1DataCollatorForMLM initialized:")
        logger.info(f"  - MLM probability: {mlm_probability}")
        logger.info(f"  - Mask token: {mask_token} (ID: {self.mask_token_id})")
        logger.info(f"  - Pad token ID: {self.pad_token_id}")
        logger.info(f"  - Vocab size: {self.vocab_size}")
        logger.info(f"  - Max total tokens: {max_total_tokens}")
        logger.info(f"  - Max query tokens: {max_query_tokens} (O(n²) self-attention)")

    def _truncate_sequence(self, sequence: str) -> str:
        """
        Truncate a multi-sequence string to fit within token limits.

        Two-stage truncation:
        1. Truncate query sequence if it exceeds max_query_tokens (critical for O(n²) attention)
        2. Remove context sequences from the beginning if total exceeds max_total_tokens

        Args:
            sequence: Comma-separated sequence string (context1,context2,...,query)

        Returns:
            Truncated sequence string
        """
        # Split into individual sequences
        sequences = sequence.split(",")

        # Each sequence gets 4 special tokens: <bos>, 1, ..., 2, <eos>
        tokens_per_seq = 4

        # Query is the last sequence - this uses O(n²) self-attention!
        query = sequences[-1]

        # Stage 1: Truncate query if it exceeds max_query_tokens
        # This is critical because query uses full self-attention (O(query_len²))
        # while context uses efficient block-causal attention
        max_query_chars = (
            self.max_query_tokens - tokens_per_seq
        )  # Account for special tokens
        if len(query) > max_query_chars:
            query = query[:max_query_chars]
            sequences[-1] = query

        if len(sequences) <= 1:
            # Single sequence (query only), return truncated query
            return query

        context_seqs = sequences[:-1]

        # Stage 2: Calculate total tokens and remove context if needed
        total_tokens = sum(len(seq) + tokens_per_seq for seq in sequences)

        if total_tokens <= self.max_total_tokens:
            return ",".join(context_seqs) + "," + query

        # Remove context sequences from the beginning until we fit
        while context_seqs and total_tokens > self.max_total_tokens:
            removed = context_seqs.pop(0)
            total_tokens -= len(removed) + tokens_per_seq

        if context_seqs:
            return ",".join(context_seqs) + "," + query
        else:
            # All context removed, return query only
            return query

    def __call__(self, examples: List[str]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            examples: List of comma-separated sequence strings

        Returns:
            Dictionary with batched tensors ready for E1 model forward pass
        """
        # Truncate sequences that exceed max_total_tokens to prevent OOM
        truncated_examples = [self._truncate_sequence(ex) for ex in examples]

        # First, prepare batches using E1BatchPreparer
        # This gives us input_ids, sequence_ids, position IDs, and initial labels
        # Use CPU device for data collation (Trainer will move to GPU later)
        batch_dict = self.batch_preparer.get_batch_kwargs(
            truncated_examples, device=torch.device("cpu")
        )

        # Extract tensors
        input_ids = batch_dict["input_ids"]  # Shape: (batch_size, seq_len)
        sequence_ids = batch_dict["sequence_ids"]  # Shape: (batch_size, seq_len)
        within_seq_position_ids = batch_dict["within_seq_position_ids"]
        global_position_ids = batch_dict["global_position_ids"]
        labels = batch_dict["labels"].clone()  # Start with copy of input_ids

        batch_size, seq_len = input_ids.shape

        # Get boundary token mask (exclude special tokens from masking)
        # Create a local copy of boundary_token_ids on the same device as input_ids
        # This avoids modifying the instance variable in DataLoader worker processes
        boundary_token_ids = self.batch_preparer.boundary_token_ids.to(input_ids.device)
        boundary_mask = torch.isin(input_ids, boundary_token_ids)
        # Shape: (batch_size, seq_len) - True for boundary tokens

        # Identify query sequence (last sequence in each multi-sequence input)
        # sequence_ids contains sequence ID for each token
        # The maximum sequence_id in each sample is the query sequence
        query_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
        for i in range(batch_size):
            max_seq_id = sequence_ids[i].max().item()
            query_mask[i] = sequence_ids[i] == max_seq_id

        # Create valid mask: only mask tokens in query sequence that are not boundary tokens
        valid_mask = query_mask & (~boundary_mask)
        # Shape: (batch_size, seq_len) - True for tokens that can be masked

        # Apply BERT-style masking
        # Step 1: Select mlm_probability of valid positions to mask
        masking_probabilities = torch.rand(batch_size, seq_len)
        selected_mask = (masking_probabilities < self.mlm_probability) & valid_mask
        # Shape: (batch_size, seq_len) - True for positions selected for masking

        # Step 2: Among selected positions, apply 80/10/10 split
        # 80% -> mask token, 10% -> random token, 10% -> unchanged
        mask_decision = torch.rand(batch_size, seq_len)

        # Create masks for each type
        mask_token_mask = (mask_decision < 0.8) & selected_mask  # 80% -> mask token
        random_token_mask = (
            (mask_decision >= 0.8) & (mask_decision < 0.9) & selected_mask
        )  # 10% -> random
        unchanged_mask = (mask_decision >= 0.9) & selected_mask  # 10% -> unchanged

        # Apply masking to input_ids
        masked_input_ids = input_ids.clone()

        # Replace with mask token (80%)
        masked_input_ids[mask_token_mask] = self.mask_token_id

        # Replace with random tokens (10%)
        # Generate random token IDs (excluding special tokens)
        # For simplicity, sample from full vocabulary, but in practice might want to exclude special tokens
        random_tokens = torch.randint(
            0,
            self.vocab_size,
            (batch_size, seq_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        masked_input_ids[random_token_mask] = random_tokens[random_token_mask]

        # 10% unchanged - no modification needed

        # Update labels: Set to original token ID at masked positions, pad_token_id elsewhere
        # All selected positions (mask_token, random_token, unchanged) should have labels set to original
        labels = torch.full_like(input_ids, self.pad_token_id)
        labels[selected_mask] = input_ids[selected_mask]

        # Note: E1BatchPreparer sets labels for context sequences to pad_token_id by default
        # (when preserve_context_labels=False), so we only need to set labels for query sequence

        return {
            "input_ids": masked_input_ids,
            "sequence_ids": sequence_ids,
            "within_seq_position_ids": within_seq_position_ids,
            "global_position_ids": global_position_ids,
            "labels": labels,
        }
