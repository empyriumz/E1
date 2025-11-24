import os
import random
import pandas as pd
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset
import logging

from E1.msa_sampling import sample_context

logger = logging.getLogger(__name__)


def validate_msa_file(msa_path: Path) -> bool:
    """
    Validate that an MSA file is readable and contains valid sequences.

    Single-sequence MSAs are considered valid and will be processed in single sequence mode.

    Args:
        msa_path: Path to MSA file

    Returns:
        True if valid (including single-sequence MSAs), False otherwise
    """
    try:
        if not msa_path.exists() or msa_path.stat().st_size == 0:
            return False

        from E1.msa_sampling import parse_msa

        msa_sequences = parse_msa(str(msa_path))

        if len(msa_sequences) == 0:
            return False

        # Check that query sequence (first sequence) is valid
        query_seq = msa_sequences[0].sequence.upper().replace("-", "").replace(".", "")
        if not query_seq or not query_seq.strip():
            return False

        # Note: Single-sequence MSAs (len == 1) are valid and will use single sequence mode
        return True
    except Exception as e:
        logger.debug(f"MSA validation failed for {msa_path}: {e}")
        return False


class E1MSADataset(Dataset):
    """
    Dataset class for E1 model that handles MSA files (.a3m format).

    For MSA files: Samples context sequences from .a3m files and returns
    comma-separated string format expected by E1.

    For single sequences: Returns the sequence string directly.
    """

    def __init__(
        self,
        msa_files: Optional[List[str]] = None,
        sequences: Optional[List[str]] = None,
        max_num_samples: int = 64,
        max_token_length: int = 16384,
        max_query_similarity: float = 0.95,
        min_query_similarity: float = 0.0,
        neighbor_similarity_lower_bound: float = 0.8,
        seed: Optional[int] = None,
        is_validation: bool = False,
    ):
        """
        Initialize E1MSADataset.

        Args:
            msa_files: List of paths to .a3m MSA files (for homologs)
            sequences: List of sequence strings (for SwissProt or single sequences)
            max_num_samples: Maximum number of sequences to sample from MSA
            max_token_length: Maximum token length for context sampling
            max_query_similarity: Maximum similarity of context sequences to query
            min_query_similarity: Minimum similarity of context sequences to query
            neighbor_similarity_lower_bound: Minimum similarity for neighbor weighting
            seed: Random seed for sampling (None = random, fixed value = deterministic)
            is_validation: If True, uses fixed seed for reproducibility
        """
        if msa_files is None and sequences is None:
            raise ValueError("Either msa_files or sequences must be provided")
        if msa_files is not None and sequences is not None:
            raise ValueError("Cannot provide both msa_files and sequences")

        self.msa_files = msa_files if msa_files is not None else []
        self.sequences = sequences if sequences is not None else []
        self.max_num_samples = max_num_samples
        self.max_token_length = max_token_length
        self.max_query_similarity = max_query_similarity
        self.min_query_similarity = min_query_similarity
        self.neighbor_similarity_lower_bound = neighbor_similarity_lower_bound
        self.is_validation = is_validation

        # Use fixed seed for validation, random for training
        if seed is None:
            self.seed = 42 if is_validation else random.randint(0, 2**31 - 1)
        else:
            self.seed = seed

        # Store base seed for epoch-based variation during training
        self.base_seed = self.seed

        logger.info(f"E1MSADataset initialized:")
        logger.info(f"  - MSA files: {len(self.msa_files)}")
        logger.info(f"  - Sequences: {len(self.sequences)}")
        logger.info(f"  - Max samples: {max_num_samples}")
        logger.info(f"  - Max token length: {max_token_length}")
        logger.info(f"  - Is validation: {is_validation}")
        logger.info(f"  - Seed: {self.seed}")

    def set_epoch(self, epoch: int):
        """
        Set epoch for dynamic sampling during training.
        Only affects training datasets, not validation.

        Args:
            epoch: Current training epoch
        """
        if not self.is_validation:
            # Vary seed by epoch for training diversity
            self.seed = self.base_seed + epoch

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        if self.msa_files:
            return len(self.msa_files)
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        """
        Get a sample from the dataset.

        For MSA files: Returns comma-separated string with context + query.
        For sequences: Returns single sequence string.

        Args:
            idx: Index of the sample

        Returns:
            Comma-separated sequence string for E1 model
        """
        if self.msa_files:
            # Handle MSA files
            msa_path = self.msa_files[idx]

            if not os.path.exists(msa_path):
                logger.error(f"MSA file not found: {msa_path}")
                raise FileNotFoundError(f"MSA file not found: {msa_path}")

            try:
                # Parse MSA to get query sequence (first sequence)
                from E1.msa_sampling import parse_msa

                msa_sequences = parse_msa(str(msa_path))
                if len(msa_sequences) == 0:
                    logger.error(f"Empty MSA file: {msa_path}")
                    raise ValueError(f"Empty MSA file: {msa_path}")

                # Extract query sequence (first sequence in MSA)
                query_seq = (
                    msa_sequences[0].sequence.upper().replace("-", "").replace(".", "")
                )

                # Validate query sequence is not empty
                if not query_seq or not query_seq.strip():
                    logger.error(f"Invalid query sequence in {msa_path}")
                    raise ValueError(f"Invalid query sequence in {msa_path}")

                # If MSA has only 1 sequence, use single sequence mode (no context sampling)
                if len(msa_sequences) == 1:
                    logger.debug(
                        f"MSA {msa_path} has only 1 sequence, using single sequence mode"
                    )
                    return query_seq

                # Sample context from MSA (excludes query, samples other sequences)
                try:
                    context_str, context_ids = sample_context(
                        msa_path=str(msa_path),
                        max_num_samples=self.max_num_samples,
                        max_token_length=self.max_token_length,
                        max_query_similarity=self.max_query_similarity,
                        min_query_similarity=self.min_query_similarity,
                        neighbor_similarity_lower_bound=self.neighbor_similarity_lower_bound,
                        seed=self.seed + idx,  # Vary seed by index for diversity
                    )

                    # Format: CONTEXT_SEQ1,CONTEXT_SEQ2,...,QUERY_SEQ
                    # If context_str is empty, just return query sequence
                    if context_str:
                        return context_str + "," + query_seq
                    else:
                        return query_seq

                except (AssertionError, ValueError) as e:
                    # If context sampling fails (e.g., no sequences within similarity range),
                    # fall back to single sequence mode
                    if "No sequences found with similarity" in str(e):
                        logger.debug(
                            f"No suitable context sequences for {msa_path}, using single sequence mode"
                        )
                        return query_seq
                    else:
                        raise

            except Exception as e:
                logger.error(f"Error processing MSA {msa_path}: {e}")
                raise

        else:
            # Handle single sequences (SwissProt)
            sequence = self.sequences[idx]
            # Validate sequence is not empty
            if not sequence or not sequence.strip():
                logger.error(f"Empty sequence at index {idx}")
                raise ValueError(f"Empty sequence at index {idx}")
            # E1 handles single sequences fine (treats as context of size 1)
            return sequence.strip()


def create_e1_datasets_from_config(
    data_conf: dict,
    general_conf: dict,
    msa_sampling_conf: Optional[dict] = None,
) -> tuple[E1MSADataset, E1MSADataset, E1MSADataset, E1MSADataset]:
    """
    Create train and validation datasets for E1 fine-tuning.

    Args:
        data_conf: Data configuration dictionary
        general_conf: General configuration dictionary (for seed)
        msa_sampling_conf: MSA sampling configuration dictionary

    Returns:
        Tuple of (homologs_train, homologs_val, swissprot_train, swissprot_val) datasets
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        raise ImportError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        )

    seed = general_conf.get("seed", 1)
    msa_sampling_conf = msa_sampling_conf or {}

    # Get MSA sampling parameters
    max_num_samples = msa_sampling_conf.get("max_num_samples", 64)
    max_query_similarity = msa_sampling_conf.get("max_query_similarity", 0.95)
    min_query_similarity = msa_sampling_conf.get("min_query_similarity", 0.0)
    neighbor_similarity_lower_bound = msa_sampling_conf.get(
        "neighbor_similarity_lower_bound", 0.8
    )
    max_token_length = data_conf.get("max_length", 16384)

    # Load homologs (MSA files)
    homologs_dir = data_conf.get("homologs_dir", None)
    homologs_msa_files = []
    if homologs_dir:
        homologs_dir = Path(homologs_dir)
        if homologs_dir.exists():
            all_msa_files = list(homologs_dir.glob("*.a3m"))
            logger.info(f"Validating {len(all_msa_files)} MSA files...")

            # Validate MSA files thoroughly before including in dataset
            homologs_msa_files = [f for f in all_msa_files if validate_msa_file(f)]

            filtered_count = len(all_msa_files) - len(homologs_msa_files)
            if filtered_count > 0:
                logger.warning(f"Filtered out {filtered_count} invalid/empty MSA files")
            logger.info(
                f"Found {len(homologs_msa_files)} valid MSA files in {homologs_dir}"
            )

            if len(homologs_msa_files) == 0:
                logger.warning(
                    "No valid MSA files found. Training will use single sequences only."
                )
        else:
            logger.warning(
                f"Homologs directory not found: {homologs_dir}. Training will use single sequences only."
            )
    else:
        logger.info(
            "No homologs_dir specified. Training will use single sequences only."
        )

    # Split homologs
    if len(homologs_msa_files) > 0:
        hom_train_files, hom_val_files = train_test_split(
            homologs_msa_files,
            test_size=data_conf.get("val_split_ratio", 0.1),
            random_state=seed,
        )
        logger.info(
            f"Creating MSA datasets: {len(hom_train_files)} train, {len(hom_val_files)} val"
        )
        homologs_train = E1MSADataset(
            msa_files=[str(f) for f in hom_train_files],
            max_num_samples=max_num_samples,
            max_token_length=max_token_length,
            max_query_similarity=max_query_similarity,
            min_query_similarity=min_query_similarity,
            neighbor_similarity_lower_bound=neighbor_similarity_lower_bound,
            seed=seed,
            is_validation=False,
        )
        homologs_val = E1MSADataset(
            msa_files=[str(f) for f in hom_val_files],
            max_num_samples=max_num_samples,
            max_token_length=max_token_length,
            max_query_similarity=max_query_similarity,
            min_query_similarity=min_query_similarity,
            neighbor_similarity_lower_bound=neighbor_similarity_lower_bound,
            seed=seed,
            is_validation=True,  # Fixed seed for validation
        )
    else:
        logger.info("No valid MSA files - creating empty MSA datasets")
        homologs_train = E1MSADataset(sequences=[])
        homologs_val = E1MSADataset(sequences=[])

    # Load SwissProt sequences
    swissprot_file = data_conf.get("swissprot_file", None)
    swissprot_sequences = []
    if swissprot_file and os.path.exists(swissprot_file):
        df = pd.read_csv(swissprot_file)
        if "sequence" in df.columns:
            # Filter out empty, None, or whitespace-only sequences
            raw_sequences = df["sequence"].tolist()
            swissprot_sequences = [
                str(seq).strip()
                for seq in raw_sequences
                if seq is not None
                and str(seq).strip()
                and str(seq).strip().replace(",", "").isalpha()
            ]
            filtered_count = len(raw_sequences) - len(swissprot_sequences)
            if filtered_count > 0:
                logger.warning(
                    f"Filtered out {filtered_count} invalid sequences from {swissprot_file}"
                )
            logger.info(
                f"Loaded {len(swissprot_sequences)} valid sequences from {swissprot_file}"
            )
        else:
            logger.warning(
                f"SwissProt file {swissprot_file} does not contain 'sequence' column"
            )
    else:
        logger.warning(f"SwissProt file not found: {swissprot_file}")

    # Split SwissProt
    if len(swissprot_sequences) > 0:
        sp_train_seqs, sp_val_seqs = train_test_split(
            swissprot_sequences,
            test_size=data_conf.get("val_split_ratio", 0.1),
            random_state=seed,
        )
        logger.info(
            f"Creating single sequence datasets: {len(sp_train_seqs)} train, {len(sp_val_seqs)} val"
        )
        swissprot_train = E1MSADataset(
            sequences=sp_train_seqs,
            seed=seed,
            is_validation=False,
        )
        swissprot_val = E1MSADataset(
            sequences=sp_val_seqs,
            seed=seed,
            is_validation=True,  # Fixed seed for validation
        )
    else:
        logger.warning("No valid single sequences found")
        swissprot_train = E1MSADataset(sequences=[])
        swissprot_val = E1MSADataset(sequences=[])

    # Summary
    total_train = len(homologs_train) + len(swissprot_train)
    total_val = len(homologs_val) + len(swissprot_val)
    logger.info("=" * 60)
    logger.info("Dataset Summary:")
    logger.info(
        f"  Training:   {len(homologs_train)} MSA + {len(swissprot_train)} single = {total_train} total"
    )
    logger.info(
        f"  Validation: {len(homologs_val)} MSA + {len(swissprot_val)} single = {total_val} total"
    )
    logger.info("=" * 60)

    if total_train == 0:
        raise ValueError(
            "No valid training data found! Check your data paths and file formats."
        )

    return homologs_train, homologs_val, swissprot_train, swissprot_val
