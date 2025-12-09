"""
Dataset for E1 residue-level metal ion binding classification.

This module provides E1BindingDataset which loads:
- FASTA files with per-residue binary binding labels
- MSA context from .a3m files for each protein

The dataset returns dictionaries compatible with E1DataCollatorForResidueClassification.
"""

import os
import random
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from torch.utils.data import Dataset
import logging

from E1.msa_sampling import sample_context

logger = logging.getLogger(__name__)


def process_binding_fasta_file(
    fasta_path: str,
) -> Tuple[List[str], List[str], List[List[int]]]:
    """
    Parse FASTA file with per-residue binding labels.

    The expected format is:
        >protein_id
        PROTEIN_SEQUENCE
        010011001...  (binary label string, same length as sequence)

    Args:
        fasta_path: Path to FASTA file

    Returns:
        Tuple of (id_list, sequence_list, label_list)
    """
    id_list, seq_list, label_list = [], [], []

    with open(fasta_path, "r") as f:
        lines = f.readlines()

    # Parse 3-line blocks: header, sequence, labels
    i = 0
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue

        # Parse header
        if lines[i].startswith(">"):
            protein_id = lines[i].strip().replace(">", "")
            i += 1

            # Parse sequence
            if i < len(lines):
                sequence = lines[i].strip().upper()
                i += 1
            else:
                logger.warning(f"Missing sequence for {protein_id}")
                continue

            # Parse labels
            if i < len(lines) and not lines[i].startswith(">"):
                label_str = lines[i].strip()
                i += 1

                # Convert to list of integers
                try:
                    labels = [int(c) for c in label_str]
                except ValueError:
                    logger.warning(
                        f"Invalid label format for {protein_id}: {label_str[:50]}..."
                    )
                    continue

                # Validate lengths match
                if len(sequence) != len(labels):
                    logger.warning(
                        f"Length mismatch for {protein_id}: seq={len(sequence)}, labels={len(labels)}"
                    )
                    continue

                id_list.append(protein_id)
                seq_list.append(sequence)
                label_list.append(labels)
            else:
                logger.warning(f"Missing labels for {protein_id}")
        else:
            i += 1

    logger.info(f"Loaded {len(id_list)} proteins from {fasta_path}")

    return id_list, seq_list, label_list


class E1BindingDataset(Dataset):
    """
    Dataset for E1 residue-level binding classification with MSA context.

    Loads:
    - FASTA files with per-residue binding labels
    - MSA context from .a3m files (optional)

    Returns dictionaries with:
    - sequence: Protein sequence string
    - labels: Per-residue binary labels
    - msa_context: Comma-separated context sequences (or None)
    """

    def __init__(
        self,
        fasta_path: str,
        msa_dir: Optional[str] = None,
        ion_type: str = "CA",
        id_list: Optional[List[str]] = None,
        max_num_samples: int = 64,
        max_token_length: int = 8192,
        max_query_similarity: float = 0.95,
        min_query_similarity: float = 0.0,
        neighbor_similarity_lower_bound: float = 0.8,
        seed: int = 42,
        is_validation: bool = False,
    ):
        """
        Initialize E1BindingDataset.

        Args:
            fasta_path: Path to FASTA file with binding labels
            msa_dir: Directory containing .a3m MSA files (one per protein ID)
            ion_type: Ion type (used for logging)
            id_list: Optional list of protein IDs to include (for CV splits)
            max_num_samples: Maximum MSA context sequences to sample
            max_token_length: Maximum total token length
            max_query_similarity: Max similarity for context sequences
            min_query_similarity: Min similarity for context sequences
            neighbor_similarity_lower_bound: Similarity threshold for weighting
            seed: Random seed for MSA sampling
            is_validation: If True, uses fixed seed for reproducibility
        """
        self.fasta_path = fasta_path
        self.msa_dir = msa_dir
        self.ion_type = ion_type
        self.max_num_samples = max_num_samples
        self.max_token_length = max_token_length
        self.max_query_similarity = max_query_similarity
        self.min_query_similarity = min_query_similarity
        self.neighbor_similarity_lower_bound = neighbor_similarity_lower_bound
        self.is_validation = is_validation

        # Set seed
        self.base_seed = seed
        self.seed = seed

        # Load FASTA data
        all_ids, all_seqs, all_labels = process_binding_fasta_file(fasta_path)

        # Create lookup dictionaries
        self._id_to_seq = {id_: seq for id_, seq in zip(all_ids, all_seqs)}
        self._id_to_labels = {id_: labels for id_, labels in zip(all_ids, all_labels)}

        # Filter to specified ID list if provided
        if id_list is not None:
            # Keep only IDs that exist in the FASTA file
            valid_ids = [id_ for id_ in id_list if id_ in self._id_to_seq]
            missing_ids = set(id_list) - set(valid_ids)
            if missing_ids:
                logger.warning(
                    f"[{ion_type}] {len(missing_ids)} IDs not found in FASTA: {list(missing_ids)[:5]}..."
                )
            self.ids = valid_ids
        else:
            self.ids = all_ids

        # Build MSA path mapping if msa_dir provided
        self._msa_paths: Dict[str, str] = {}
        if msa_dir and os.path.isdir(msa_dir):
            for protein_id in self.ids:
                msa_path = Path(msa_dir) / f"{protein_id}.a3m"
                if msa_path.exists():
                    self._msa_paths[protein_id] = str(msa_path)

        # Stats
        self.num_with_msa = len(self._msa_paths)
        self.num_without_msa = len(self.ids) - self.num_with_msa

        logger.info(f"E1BindingDataset [{ion_type}] initialized:")
        logger.info(f"  - FASTA: {fasta_path}")
        logger.info(f"  - Proteins: {len(self.ids)}")
        logger.info(f"  - With MSA: {self.num_with_msa}")
        logger.info(f"  - Without MSA: {self.num_without_msa}")
        logger.info(f"  - Is validation: {is_validation}")

    def set_epoch(self, epoch: int):
        """
        Set epoch for dynamic MSA sampling during training.

        Args:
            epoch: Current training epoch
        """
        if not self.is_validation:
            self.seed = self.base_seed + epoch
            logger.debug(
                f"E1BindingDataset [{self.ion_type}] set_epoch({epoch}), seed={self.seed}"
            )

    def __len__(self) -> int:
        return len(self.ids)

    def _sample_msa_context(self, protein_id: str, idx: int) -> Optional[str]:
        """
        Sample MSA context for a protein if available.

        Args:
            protein_id: Protein identifier
            idx: Dataset index (used for seed variation)

        Returns:
            Comma-separated context sequences, or None if no MSA available
        """
        if protein_id not in self._msa_paths:
            return None

        msa_path = self._msa_paths[protein_id]

        try:
            # Use sample_context from E1.msa_sampling
            context_str, _ = sample_context(
                msa_path=msa_path,
                max_num_samples=self.max_num_samples,
                max_token_length=self.max_token_length,
                max_query_similarity=self.max_query_similarity,
                min_query_similarity=self.min_query_similarity,
                neighbor_similarity_lower_bound=self.neighbor_similarity_lower_bound,
                seed=self.seed + idx,  # Vary seed by index
                device=torch.device("cpu"),  # Use CPU to avoid CUDA issues in workers
            )
            return context_str
        except Exception as e:
            logger.warning(f"Failed to sample MSA for {protein_id}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with:
            - sequence: Protein sequence string
            - labels: Per-residue binary labels (List[int])
            - msa_context: Comma-separated context sequences (or None)
            - protein_id: Protein identifier
        """
        protein_id = self.ids[idx]
        sequence = self._id_to_seq[protein_id]
        labels = self._id_to_labels[protein_id]

        # Sample MSA context if available
        msa_context = self._sample_msa_context(protein_id, idx)

        return {
            "sequence": sequence,
            "labels": labels,
            "msa_context": msa_context,
            "protein_id": protein_id,
        }

    def get_class_counts(self) -> tuple:
        """
        Get total positive and negative residue counts.

        Returns:
            Tuple of (total_positive, total_negative)
        """
        total_pos = 0
        total_neg = 0

        for protein_id in self.ids:
            labels = self._id_to_labels[protein_id]
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            total_pos += pos_count
            total_neg += neg_count

        logger.info(
            f"E1BindingDataset [{self.ion_type}] class counts: "
            f"pos={total_pos}, neg={total_neg}, ratio={total_neg/max(total_pos,1):.1f}:1"
        )

        return total_pos, total_neg


class E1BindingDatasetREE(Dataset):
    """
    Dataset that combines LREE and HREE datasets to create REE dataset.

    This dataset combines all sequences from both LREE and HREE datasets:
    - For proteins present in both datasets: combine labels using logical OR
    - For proteins only in one dataset: use those labels as-is
    - All sequences from both datasets are included (union of protein IDs)
    """

    def __init__(
        self,
        lree_dataset: E1BindingDataset,
        hree_dataset: E1BindingDataset,
    ):
        """
        Initialize REE dataset from LREE and HREE datasets.

        Args:
            lree_dataset: E1BindingDataset for LREE
            hree_dataset: E1BindingDataset for HREE
        """
        self.lree_dataset = lree_dataset
        self.hree_dataset = hree_dataset
        self.ion_type = "REE"

        # Get union of all protein IDs from both datasets
        lree_ids_set = set(lree_dataset.ids)
        hree_ids_set = set(hree_dataset.ids)
        all_ids = sorted(list(lree_ids_set | hree_ids_set))
        common_ids = sorted(list(lree_ids_set & hree_ids_set))

        self.ids = all_ids

        logger.info(
            f"REE dataset: Combining LREE and HREE datasets. "
            f"LREE: {len(lree_dataset.ids)} proteins, "
            f"HREE: {len(hree_dataset.ids)} proteins, "
            f"Total (union): {len(all_ids)} proteins, "
            f"Common: {len(common_ids)} proteins"
        )

        # Build combined dataset: sequences and labels
        self._id_to_seq = {}
        self._id_to_labels = {}

        for protein_id in self.ids:
            lree_labels = lree_dataset._id_to_labels.get(protein_id)
            hree_labels = hree_dataset._id_to_labels.get(protein_id)
            lree_seq = lree_dataset._id_to_seq.get(protein_id)
            hree_seq = hree_dataset._id_to_seq.get(protein_id)

            # Get sequence (prefer LREE if both exist, but they should match)
            if lree_seq is not None:
                sequence = lree_seq
            elif hree_seq is not None:
                sequence = hree_seq
            else:
                logger.warning(f"REE dataset: No sequence found for {protein_id}")
                continue

            # Warn if sequences don't match (shouldn't happen for same protein)
            if lree_seq is not None and hree_seq is not None and lree_seq != hree_seq:
                logger.warning(
                    f"REE dataset: Sequence mismatch for {protein_id} "
                    f"(lengths: LREE={len(lree_seq)}, HREE={len(hree_seq)})"
                )

            # Combine labels
            if lree_labels is not None and hree_labels is not None:
                # Protein in both datasets: combine labels with logical OR
                if len(lree_labels) != len(hree_labels):
                    logger.warning(
                        f"REE dataset: Label length mismatch for {protein_id}: "
                        f"LREE={len(lree_labels)}, HREE={len(hree_labels)}. "
                        f"Using minimum length."
                    )
                    min_len = min(len(lree_labels), len(hree_labels))
                    lree_labels = lree_labels[:min_len]
                    hree_labels = hree_labels[:min_len]

                # Combine labels: REE = LREE OR HREE
                ree_labels = [l | h for l, h in zip(lree_labels, hree_labels)]
            elif lree_labels is not None:
                # Protein only in LREE: use LREE labels
                ree_labels = lree_labels
            elif hree_labels is not None:
                # Protein only in HREE: use HREE labels
                ree_labels = hree_labels
            else:
                logger.warning(f"REE dataset: No labels found for {protein_id}")
                continue

            self._id_to_seq[protein_id] = sequence
            self._id_to_labels[protein_id] = ree_labels

        # Combine MSA paths from both datasets
        self._msa_paths: Dict[str, str] = {}
        for protein_id in self.ids:
            if protein_id in lree_dataset._msa_paths:
                self._msa_paths[protein_id] = lree_dataset._msa_paths[protein_id]
            elif protein_id in hree_dataset._msa_paths:
                self._msa_paths[protein_id] = hree_dataset._msa_paths[protein_id]

        # Copy MSA sampling parameters (use LREE as default, should be same)
        self.msa_dir = (
            lree_dataset.msa_dir if lree_dataset.msa_dir else hree_dataset.msa_dir
        )
        self.max_num_samples = lree_dataset.max_num_samples
        self.max_token_length = lree_dataset.max_token_length
        self.max_query_similarity = lree_dataset.max_query_similarity
        self.min_query_similarity = lree_dataset.min_query_similarity
        self.neighbor_similarity_lower_bound = (
            lree_dataset.neighbor_similarity_lower_bound
        )
        self.is_validation = lree_dataset.is_validation
        self.base_seed = lree_dataset.base_seed
        self.seed = lree_dataset.seed

        # Stats
        self.num_with_msa = len(self._msa_paths)
        self.num_without_msa = len(self.ids) - self.num_with_msa

        logger.info(f"E1BindingDatasetREE initialized:")
        logger.info(f"  - Proteins: {len(self.ids)}")
        logger.info(f"  - With MSA: {self.num_with_msa}")
        logger.info(f"  - Without MSA: {self.num_without_msa}")
        logger.info(f"  - Is validation: {self.is_validation}")

    def set_epoch(self, epoch: int):
        """Set epoch for dynamic MSA sampling during training."""
        self.lree_dataset.set_epoch(epoch)
        self.hree_dataset.set_epoch(epoch)
        if not self.is_validation:
            self.seed = self.base_seed + epoch
            logger.debug(f"E1BindingDatasetREE set_epoch({epoch}), seed={self.seed}")

    def __len__(self) -> int:
        return len(self.ids)

    def _sample_msa_context(self, protein_id: str, idx: int) -> Optional[str]:
        """Sample MSA context for a protein if available."""
        if protein_id not in self._msa_paths:
            return None

        msa_path = self._msa_paths[protein_id]

        try:
            from E1.msa_sampling import sample_context

            context_str, _ = sample_context(
                msa_path=msa_path,
                max_num_samples=self.max_num_samples,
                max_token_length=self.max_token_length,
                max_query_similarity=self.max_query_similarity,
                min_query_similarity=self.min_query_similarity,
                neighbor_similarity_lower_bound=self.neighbor_similarity_lower_bound,
                seed=self.seed + idx,
                device=torch.device("cpu"),
            )
            return context_str
        except Exception as e:
            logger.warning(f"Failed to sample MSA for {protein_id}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        protein_id = self.ids[idx]
        sequence = self._id_to_seq[protein_id]
        labels = self._id_to_labels[protein_id]

        # Sample MSA context if available
        msa_context = self._sample_msa_context(protein_id, idx)

        return {
            "sequence": sequence,
            "labels": labels,
            "msa_context": msa_context,
            "protein_id": protein_id,
        }

    def get_class_counts(self) -> tuple:
        """Get total positive and negative residue counts from combined REE labels."""
        total_pos = 0
        total_neg = 0

        for protein_id in self.ids:
            labels = self._id_to_labels[protein_id]
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            total_pos += pos_count
            total_neg += neg_count

        logger.info(
            f"E1BindingDatasetREE class counts: "
            f"pos={total_pos}, neg={total_neg}, ratio={total_neg/max(total_pos,1):.1f}:1"
        )

        return total_pos, total_neg


class ConcatE1BindingDataset(Dataset):
    """
    Concatenates multiple E1BindingDatasets.

    Useful for combining datasets from different ions or training multiple ions together.
    """

    def __init__(
        self,
        datasets: List[E1BindingDataset],
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Initialize concatenated dataset.

        Args:
            datasets: List of E1BindingDataset instances
            shuffle: Whether to shuffle the combined indices
            seed: Random seed for shuffling
        """
        self.datasets = datasets
        self.shuffle = shuffle
        self.seed = seed

        # Build index mapping
        self._index_mapping: List[Tuple[int, int]] = []  # (dataset_idx, local_idx)
        for ds_idx, ds in enumerate(datasets):
            for local_idx in range(len(ds)):
                self._index_mapping.append((ds_idx, local_idx))

        # Shuffle if requested
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._index_mapping)

        logger.info(
            f"ConcatE1BindingDataset: {len(self._index_mapping)} samples from {len(datasets)} datasets"
        )

    def set_epoch(self, epoch: int):
        """Update epoch for all underlying datasets."""
        for ds in self.datasets:
            ds.set_epoch(epoch)

        # Re-shuffle with different seed each epoch
        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            rng.shuffle(self._index_mapping)

    def __len__(self) -> int:
        return len(self._index_mapping)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx = self._index_mapping[idx]
        return self.datasets[ds_idx][local_idx]


def get_train_val_split(
    id_list: List[str],
    fold_to_use_as_val: int = 1,
    num_folds: int = 5,
    random_seed: int = 1,
) -> Tuple[List[str], List[str]]:
    """
    Split ID list into train and validation sets using K-fold.

    Args:
        id_list: List of protein IDs
        fold_to_use_as_val: Which fold to use as validation (1-indexed)
        num_folds: Total number of folds
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_ids, val_ids)
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)

    # Convert to numpy array for sklearn
    import numpy as np

    id_array = np.array(id_list)

    # Get the specified fold
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(id_array), 1):
        if fold_idx == fold_to_use_as_val:
            train_ids = id_array[train_idx].tolist()
            val_ids = id_array[val_idx].tolist()
            return train_ids, val_ids

    raise ValueError(f"Invalid fold_to_use_as_val: {fold_to_use_as_val}")


def compute_shared_ree_splits(
    data_conf: Dict[str, Any],
    training_conf: Dict[str, Any],
    fold_to_use_as_val: int = 1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Compute shared train/val splits for REE-related ions (LREE, HREE, REE).

    This function ensures that all three datasets use the same train/val split
    based on the union of all protein IDs from LREE and HREE, preventing data
    leakage where a protein could be in REE validation but LREE/HREE training.

    Args:
        data_conf: Data configuration dictionary
        training_conf: Training configuration dictionary
        fold_to_use_as_val: Which fold to use for validation
        seed: Random seed

    Returns:
        Tuple of (train_ids, val_ids) based on the union of LREE and HREE proteins
    """
    fasta_path = data_conf.get("fasta_path", "")

    # Build FASTA paths for LREE and HREE
    if "{ION}" in fasta_path:
        lree_fasta_path = fasta_path.replace("{ION}", "LREE")
        hree_fasta_path = fasta_path.replace("{ION}", "HREE")
    else:
        lree_fasta_path = f"{fasta_path}/LREE_train.fasta"
        hree_fasta_path = f"{fasta_path}/HREE_train.fasta"

    # Load all protein IDs from both datasets
    lree_ids, _, _ = process_binding_fasta_file(lree_fasta_path)
    hree_ids, _, _ = process_binding_fasta_file(hree_fasta_path)

    # Compute union of all protein IDs
    all_ids = sorted(list(set(lree_ids) | set(hree_ids)))

    logger.info(
        f"Computing shared REE splits: LREE={len(lree_ids)}, HREE={len(hree_ids)}, "
        f"Union={len(all_ids)}"
    )

    # Apply K-fold split to the union
    num_folds = training_conf.get("num_folds", 5)
    train_ids, val_ids = get_train_val_split(
        all_ids,
        fold_to_use_as_val=fold_to_use_as_val,
        num_folds=num_folds,
        random_seed=seed,
    )

    return train_ids, val_ids


def create_binding_datasets_from_config(
    data_conf: Dict[str, Any],
    training_conf: Dict[str, Any],
    ion_type: str,
    fold_to_use_as_val: int = 1,
    seed: int = 42,
    train_ids: Optional[List[str]] = None,
    val_ids: Optional[List[str]] = None,
) -> Tuple[E1BindingDataset, E1BindingDataset, tuple]:
    """
    Create train and validation datasets from configuration.

    Args:
        data_conf: Data configuration dictionary
        training_conf: Training configuration dictionary
        ion_type: Ion type (e.g., "CA", "ZN", "REE")
        fold_to_use_as_val: Which fold to use for validation
        seed: Random seed
        train_ids: Optional pre-computed train IDs
        val_ids: Optional pre-computed validation IDs

    Returns:
        Tuple of (train_dataset, val_dataset, (pos_count, neg_count))
    """
    # Special handling for REE: combine LREE and HREE datasets
    if ion_type == "REE":
        logger.info("Creating REE dataset by combining LREE and HREE datasets")

        # Load all LREE and HREE data first (no filtering)
        lree_all, _, _ = create_binding_datasets_from_config(
            data_conf=data_conf,
            training_conf=training_conf,
            ion_type="LREE",
            fold_to_use_as_val=fold_to_use_as_val,
            seed=seed,
            train_ids=None,  # Load all data
            val_ids=None,
        )

        hree_all, _, _ = create_binding_datasets_from_config(
            data_conf=data_conf,
            training_conf=training_conf,
            ion_type="HREE",
            fold_to_use_as_val=fold_to_use_as_val,
            seed=seed,
            train_ids=None,  # Load all data
            val_ids=None,
        )

        # Combine all LREE and HREE data
        combined_all = E1BindingDatasetREE(lree_all, hree_all)

        # Apply train/val split to the combined dataset
        if train_ids is None or val_ids is None:
            all_ids = combined_all.ids
            num_folds = training_conf.get("num_folds", 5)
            train_ids, val_ids = get_train_val_split(
                all_ids,
                fold_to_use_as_val=fold_to_use_as_val,
                num_folds=num_folds,
                random_seed=seed,
            )

        # Create train and val datasets by filtering the combined dataset
        # We need to create new datasets with filtered IDs
        # Since E1BindingDatasetREE doesn't support id_list filtering directly,
        # we'll create filtered versions by creating new datasets with only the relevant IDs

        # Get MSA sampling config for TRAINING
        msa_config = training_conf.get("msa_sampling", {})
        max_num_samples = msa_config.get("max_num_samples", 64)
        max_token_length = msa_config.get("max_token_length", 8192)
        max_query_similarity = msa_config.get("max_query_similarity", 0.95)
        min_query_similarity = msa_config.get("min_query_similarity", 0.0)
        neighbor_similarity_lower_bound = msa_config.get(
            "neighbor_similarity_lower_bound", 0.8
        )

        # Get MSA sampling config for VALIDATION (fall back to training config if not specified)
        val_msa_config = training_conf.get("validation_msa_sampling", msa_config)
        val_max_num_samples = val_msa_config.get("max_num_samples", max_num_samples)
        val_max_token_length = val_msa_config.get("max_token_length", max_token_length)
        val_max_query_similarity = val_msa_config.get(
            "max_query_similarity", max_query_similarity
        )
        val_min_query_similarity = val_msa_config.get(
            "min_query_similarity", min_query_similarity
        )
        val_neighbor_similarity_lower_bound = val_msa_config.get(
            "neighbor_similarity_lower_bound", neighbor_similarity_lower_bound
        )

        # Create filtered LREE and HREE datasets for train
        fasta_path = data_conf.get("fasta_path", "")
        msa_dir = data_conf.get("msa_dir", None)

        if "{ION}" in fasta_path:
            lree_fasta_path = fasta_path.replace("{ION}", "LREE")
            hree_fasta_path = fasta_path.replace("{ION}", "HREE")
        else:
            lree_fasta_path = f"{fasta_path}/LREE_train.fasta"
            hree_fasta_path = f"{fasta_path}/HREE_train.fasta"

        if msa_dir and "{ION}" in msa_dir:
            lree_msa_dir = msa_dir.replace("{ION}", "LREE")
            hree_msa_dir = msa_dir.replace("{ION}", "HREE")
        elif msa_dir:
            lree_msa_dir = f"{msa_dir}/LREE"
            hree_msa_dir = f"{msa_dir}/HREE"
        else:
            lree_msa_dir = None
            hree_msa_dir = None

        # Create train datasets with training MSA config
        lree_train = E1BindingDataset(
            fasta_path=lree_fasta_path,
            msa_dir=lree_msa_dir,
            ion_type="LREE",
            id_list=train_ids,
            max_num_samples=max_num_samples,
            max_token_length=max_token_length,
            max_query_similarity=max_query_similarity,
            min_query_similarity=min_query_similarity,
            neighbor_similarity_lower_bound=neighbor_similarity_lower_bound,
            seed=seed,
            is_validation=False,
        )

        hree_train = E1BindingDataset(
            fasta_path=hree_fasta_path,
            msa_dir=hree_msa_dir,
            ion_type="HREE",
            id_list=train_ids,
            max_num_samples=max_num_samples,
            max_token_length=max_token_length,
            max_query_similarity=max_query_similarity,
            min_query_similarity=min_query_similarity,
            neighbor_similarity_lower_bound=neighbor_similarity_lower_bound,
            seed=seed,
            is_validation=False,
        )

        # Create val datasets with validation MSA config
        lree_val = E1BindingDataset(
            fasta_path=lree_fasta_path,
            msa_dir=lree_msa_dir,
            ion_type="LREE",
            id_list=val_ids,
            max_num_samples=val_max_num_samples,
            max_token_length=val_max_token_length,
            max_query_similarity=val_max_query_similarity,
            min_query_similarity=val_min_query_similarity,
            neighbor_similarity_lower_bound=val_neighbor_similarity_lower_bound,
            seed=seed,
            is_validation=True,
        )

        hree_val = E1BindingDataset(
            fasta_path=hree_fasta_path,
            msa_dir=hree_msa_dir,
            ion_type="HREE",
            id_list=val_ids,
            max_num_samples=val_max_num_samples,
            max_token_length=val_max_token_length,
            max_query_similarity=val_max_query_similarity,
            min_query_similarity=val_min_query_similarity,
            neighbor_similarity_lower_bound=val_neighbor_similarity_lower_bound,
            seed=seed,
            is_validation=True,
        )

        # Create REE datasets by combining filtered LREE and HREE
        train_dataset = E1BindingDatasetREE(lree_train, hree_train)
        val_dataset = E1BindingDatasetREE(lree_val, hree_val)

        # Get class counts from training set for pos_weight computation
        pos_count, neg_count = train_dataset.get_class_counts()

        return train_dataset, val_dataset, (pos_count, neg_count)

    # Standard handling for other ion types
    fasta_path = data_conf.get("fasta_path", "")
    msa_dir = data_conf.get("msa_dir", None)

    # Get MSA sampling config for TRAINING
    msa_config = training_conf.get("msa_sampling", {})
    max_num_samples = msa_config.get("max_num_samples", 64)
    max_token_length = msa_config.get("max_token_length", 8192)
    max_query_similarity = msa_config.get("max_query_similarity", 0.95)
    min_query_similarity = msa_config.get("min_query_similarity", 0.0)
    neighbor_similarity_lower_bound = msa_config.get(
        "neighbor_similarity_lower_bound", 0.8
    )

    # Get MSA sampling config for VALIDATION (fall back to training config if not specified)
    val_msa_config = training_conf.get("validation_msa_sampling", msa_config)
    val_max_num_samples = val_msa_config.get("max_num_samples", max_num_samples)
    val_max_token_length = val_msa_config.get("max_token_length", max_token_length)
    val_max_query_similarity = val_msa_config.get(
        "max_query_similarity", max_query_similarity
    )
    val_min_query_similarity = val_msa_config.get(
        "min_query_similarity", min_query_similarity
    )
    val_neighbor_similarity_lower_bound = val_msa_config.get(
        "neighbor_similarity_lower_bound", neighbor_similarity_lower_bound
    )

    # Build FASTA path
    if "{ION}" in fasta_path:
        ion_fasta_path = fasta_path.replace("{ION}", ion_type)
    else:
        ion_fasta_path = f"{fasta_path}/{ion_type}_train.fasta"

    # Build MSA dir
    if msa_dir and "{ION}" in msa_dir:
        ion_msa_dir = msa_dir.replace("{ION}", ion_type)
    elif msa_dir:
        ion_msa_dir = f"{msa_dir}/{ion_type}"
    else:
        ion_msa_dir = None

    # Load IDs from FASTA if not provided
    if train_ids is None or val_ids is None:
        all_ids, _, _ = process_binding_fasta_file(ion_fasta_path)
        num_folds = training_conf.get("num_folds", 5)
        train_ids, val_ids = get_train_val_split(
            all_ids,
            fold_to_use_as_val=fold_to_use_as_val,
            num_folds=num_folds,
            random_seed=seed,
        )

    # Create training dataset with training MSA config
    train_dataset = E1BindingDataset(
        fasta_path=ion_fasta_path,
        msa_dir=ion_msa_dir,
        ion_type=ion_type,
        id_list=train_ids,
        max_num_samples=max_num_samples,
        max_token_length=max_token_length,
        max_query_similarity=max_query_similarity,
        min_query_similarity=min_query_similarity,
        neighbor_similarity_lower_bound=neighbor_similarity_lower_bound,
        seed=seed,
        is_validation=False,
    )

    # Create validation dataset with validation MSA config
    val_dataset = E1BindingDataset(
        fasta_path=ion_fasta_path,
        msa_dir=ion_msa_dir,
        ion_type=ion_type,
        id_list=val_ids,
        max_num_samples=val_max_num_samples,
        max_token_length=val_max_token_length,
        max_query_similarity=val_max_query_similarity,
        min_query_similarity=val_min_query_similarity,
        neighbor_similarity_lower_bound=val_neighbor_similarity_lower_bound,
        seed=seed,
        is_validation=True,  # Fixed seed for validation
    )

    # Log MSA config differences if they exist
    if val_msa_config != msa_config:
        logger.info(f"Using separate MSA sampling configs for train/val:")
        logger.info(
            f"  Train: max_num_samples={max_num_samples}, max_token_length={max_token_length}"
        )
        logger.info(
            f"  Val:   max_num_samples={val_max_num_samples}, max_token_length={val_max_token_length}"
        )

    # Get class counts from training set for pos_weight computation
    pos_count, neg_count = train_dataset.get_class_counts()

    return train_dataset, val_dataset, (pos_count, neg_count)
