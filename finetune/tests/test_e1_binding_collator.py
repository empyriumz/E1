"""
Tests for E1DataCollatorForResidueClassification (binding collator).

Tests verify:
- Output structure and tensor shapes
- Label alignment with residue tokens
- Special token handling (ignore_index)
- Sequence truncation
- MSA context handling
- Label smoothing
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

# Add finetune directory to path
finetune_dir = Path(__file__).parent.parent
if str(finetune_dir) not in sys.path:
    sys.path.insert(0, str(finetune_dir))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_e1_batch_preparer():
    """Create a mock E1BatchPreparer for testing."""
    mock_preparer = MagicMock()
    mock_preparer.pad_token_id = 0
    mock_preparer.boundary_token_ids = torch.tensor(
        [0, 27, 28, 29, 30]
    )  # pad, bos, eos, 1, 2

    def mock_get_batch_kwargs(sequences, device=torch.device("cpu")):
        """Generate mock batch dictionaries with proper structure."""
        batch_size = len(sequences)
        # Calculate max length accounting for special tokens
        max_len = 0
        for seq in sequences:
            parts = seq.split(",")
            query = parts[-1]
            # Each sequence gets: <bos> 1 [residues...] 2 <eos>
            seq_tokens = 4 + len(query)  # 4 special tokens
            max_len = max(max_len, seq_tokens)

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        sequence_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        within_seq_position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        global_position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, seq in enumerate(sequences):
            parts = seq.split(",")
            query = parts[-1]
            # Tokenize: <bos>=27, 1=29, [amino acids]=1-26, 2=30, <eos>=28
            tokens = [27, 29]  # <bos> 1
            for c in query:
                tokens.append(ord(c) - ord("A") + 1)  # A=1, B=2, etc.
            tokens.extend([30, 28])  # 2 <eos>

            seq_len = len(tokens)
            input_ids[i, :seq_len] = torch.tensor(tokens)
            # Query sequence gets max sequence_id
            sequence_ids[i, :seq_len] = len(parts) - 1
            within_seq_position_ids[i, :seq_len] = torch.arange(seq_len)
            global_position_ids[i, :seq_len] = torch.arange(seq_len)

        return {
            "input_ids": input_ids.to(device),
            "sequence_ids": sequence_ids.to(device),
            "within_seq_position_ids": within_seq_position_ids.to(device),
            "global_position_ids": global_position_ids.to(device),
        }

    mock_preparer.get_batch_kwargs = mock_get_batch_kwargs
    return mock_preparer


@pytest.fixture
def binding_collator(mock_e1_batch_preparer):
    """Create collator with mocked batch preparer."""
    with patch(
        "training.e1_binding_collator.E1BatchPreparer",
        return_value=mock_e1_batch_preparer,
    ):
        from training.e1_binding_collator import E1DataCollatorForResidueClassification

        collator = E1DataCollatorForResidueClassification(
            max_total_tokens=8192,
            max_query_tokens=1024,
            ignore_index=-100,
            label_smoothing=0.0,
        )
        # Replace batch_preparer with our mock
        collator.batch_preparer = mock_e1_batch_preparer
        collator.pad_token_id = mock_e1_batch_preparer.pad_token_id
        collator.boundary_token_ids = mock_e1_batch_preparer.boundary_token_ids
        return collator


@pytest.fixture
def simple_examples():
    """Simple examples for basic tests."""
    return [
        {
            "sequence": "ACDEF",
            "labels": [0, 1, 0, 1, 0],
            "msa_context": None,
            "protein_id": "protein_0",
        },
        {
            "sequence": "GHIK",
            "labels": [1, 0, 1, 0],
            "msa_context": None,
            "protein_id": "protein_1",
        },
    ]


# =============================================================================
# Output Structure Tests
# =============================================================================


class TestCollatorOutputStructure:
    """Tests for collator output structure."""

    def test_output_has_required_keys(self, binding_collator, simple_examples):
        """Verify all expected keys are present in output."""
        batch = binding_collator(simple_examples)

        required_keys = [
            "input_ids",
            "sequence_ids",
            "within_seq_position_ids",
            "global_position_ids",
            "binding_labels",
            "label_mask",
            "protein_ids",
        ]

        for key in required_keys:
            assert key in batch, f"Missing key: {key}"

    def test_output_tensors_are_batched(self, binding_collator, simple_examples):
        """Verify output tensors have batch dimension."""
        batch = binding_collator(simple_examples)
        batch_size = len(simple_examples)

        assert batch["input_ids"].shape[0] == batch_size
        assert batch["binding_labels"].shape[0] == batch_size
        assert batch["label_mask"].shape[0] == batch_size

    def test_protein_ids_preserved(self, binding_collator, simple_examples):
        """Verify protein IDs are preserved in output."""
        batch = binding_collator(simple_examples)

        assert batch["protein_ids"] == ["protein_0", "protein_1"]


# =============================================================================
# Label Alignment Tests
# =============================================================================


class TestLabelAlignment:
    """Tests for label alignment with residue tokens."""

    def test_label_count_matches_residue_count(self, binding_collator, simple_examples):
        """Number of valid labels should match number of residues."""
        batch = binding_collator(simple_examples)

        for i, example in enumerate(simple_examples):
            expected_residues = len(example["sequence"])
            actual_valid = batch["label_mask"][i].sum().item()
            assert (
                actual_valid == expected_residues
            ), f"Example {i}: expected {expected_residues} valid labels, got {actual_valid}"

    def test_labels_at_valid_positions(self, binding_collator, simple_examples):
        """Labels should only be set at valid (non-ignore) positions."""
        batch = binding_collator(simple_examples)
        ignore_index = -100

        for i in range(len(simple_examples)):
            labels = batch["binding_labels"][i]
            mask = batch["label_mask"][i]

            # Check that labels at invalid positions are ignore_index
            invalid_labels = labels[~mask]
            assert (
                invalid_labels == ignore_index
            ).all(), f"Example {i}: invalid positions should have ignore_index"


# =============================================================================
# Special Token Tests
# =============================================================================


class TestSpecialTokenHandling:
    """Tests for special token handling."""

    def test_special_tokens_have_ignore_index(self, binding_collator):
        """Special tokens (<bos>, <eos>, 1, 2) should have ignore_index labels."""
        examples = [{"sequence": "ABC", "labels": [0, 1, 0], "protein_id": "test"}]
        batch = binding_collator(examples)

        # First two tokens are <bos> and 1, last two are 2 and <eos>
        # These should not be in the label_mask
        label_mask = batch["label_mask"][0]

        # Count valid positions - should be exactly 3 (A, B, C)
        valid_count = label_mask.sum().item()
        assert valid_count == 3, f"Expected 3 valid positions, got {valid_count}"


# =============================================================================
# Truncation Tests
# =============================================================================


class TestSequenceTruncation:
    """Tests for sequence truncation behavior."""

    def test_long_sequence_truncated(self, mock_e1_batch_preparer):
        """Sequences exceeding max_query_tokens should be truncated."""
        with patch(
            "training.e1_binding_collator.E1BatchPreparer",
            return_value=mock_e1_batch_preparer,
        ):
            from training.e1_binding_collator import (
                E1DataCollatorForResidueClassification,
            )

            collator = E1DataCollatorForResidueClassification(
                max_total_tokens=100,
                max_query_tokens=10,  # Very short for testing
                ignore_index=-100,
            )
            collator.batch_preparer = mock_e1_batch_preparer
            collator.pad_token_id = mock_e1_batch_preparer.pad_token_id
            collator.boundary_token_ids = mock_e1_batch_preparer.boundary_token_ids

            # Create example longer than max_query_tokens
            long_seq = "A" * 20
            long_labels = [0] * 20
            examples = [
                {"sequence": long_seq, "labels": long_labels, "protein_id": "test"}
            ]

            batch = collator(examples)

            # Verify truncation occurred
            # max_query_tokens=10, minus 4 special tokens = 6 max residues
            valid_count = batch["label_mask"][0].sum().item()
            assert (
                valid_count <= 10 - 4
            ), f"Sequence should be truncated, got {valid_count} valid positions"


# =============================================================================
# MSA Context Tests
# =============================================================================


class TestMSAContextHandling:
    """Tests for MSA context handling."""

    def test_context_labels_ignored(self, mock_e1_batch_preparer):
        """Context sequence residues should have ignore_index labels."""
        with patch(
            "training.e1_binding_collator.E1BatchPreparer",
            return_value=mock_e1_batch_preparer,
        ):
            from training.e1_binding_collator import (
                E1DataCollatorForResidueClassification,
            )

            collator = E1DataCollatorForResidueClassification(
                max_total_tokens=8192,
                max_query_tokens=1024,
                ignore_index=-100,
            )
            collator.batch_preparer = mock_e1_batch_preparer
            collator.pad_token_id = mock_e1_batch_preparer.pad_token_id
            collator.boundary_token_ids = mock_e1_batch_preparer.boundary_token_ids

            # Example with context
            examples = [
                {
                    "sequence": "ABC",
                    "labels": [0, 1, 0],
                    "msa_context": "DEF",  # Context sequence
                    "protein_id": "test",
                }
            ]

            batch = collator(examples)

            # Only query residues should be valid
            valid_count = batch["label_mask"][0].sum().item()
            assert (
                valid_count == 3
            ), f"Only 3 query residues should be valid, got {valid_count}"


# =============================================================================
# Label Smoothing Tests
# =============================================================================


class TestLabelSmoothing:
    """Tests for label smoothing functionality."""

    def test_label_smoothing_applied(self, mock_e1_batch_preparer):
        """Label smoothing should modify label values."""
        with patch(
            "training.e1_binding_collator.E1BatchPreparer",
            return_value=mock_e1_batch_preparer,
        ):
            from training.e1_binding_collator import (
                E1DataCollatorForResidueClassification,
            )

            smoothing = 0.1
            collator = E1DataCollatorForResidueClassification(
                max_total_tokens=8192,
                max_query_tokens=1024,
                ignore_index=-100,
                label_smoothing=smoothing,
            )
            collator.batch_preparer = mock_e1_batch_preparer
            collator.pad_token_id = mock_e1_batch_preparer.pad_token_id
            collator.boundary_token_ids = mock_e1_batch_preparer.boundary_token_ids

            examples = [
                {
                    "sequence": "AB",
                    "labels": [0, 1],
                    "protein_id": "test",
                }
            ]

            batch = collator(examples)
            labels = batch["binding_labels"][0]
            mask = batch["label_mask"][0]
            valid_labels = labels[mask]

            # Label 0 should become smoothing (0.1)
            # Label 1 should become 1 - smoothing (0.9)
            assert valid_labels.min().item() == pytest.approx(smoothing, abs=0.01)
            assert valid_labels.max().item() == pytest.approx(1 - smoothing, abs=0.01)

    def test_no_smoothing_when_zero(self, mock_e1_batch_preparer):
        """Labels should be 0/1 when smoothing is 0."""
        with patch(
            "training.e1_binding_collator.E1BatchPreparer",
            return_value=mock_e1_batch_preparer,
        ):
            from training.e1_binding_collator import (
                E1DataCollatorForResidueClassification,
            )

            collator = E1DataCollatorForResidueClassification(
                max_total_tokens=8192,
                max_query_tokens=1024,
                ignore_index=-100,
                label_smoothing=0.0,
            )
            collator.batch_preparer = mock_e1_batch_preparer
            collator.pad_token_id = mock_e1_batch_preparer.pad_token_id
            collator.boundary_token_ids = mock_e1_batch_preparer.boundary_token_ids

            examples = [
                {
                    "sequence": "AB",
                    "labels": [0, 1],
                    "protein_id": "test",
                }
            ]

            batch = collator(examples)
            labels = batch["binding_labels"][0]
            mask = batch["label_mask"][0]
            valid_labels = labels[mask]

            # Labels should be exactly 0 and 1
            assert 0.0 in valid_labels.tolist()
            assert 1.0 in valid_labels.tolist()


# =============================================================================
# Label Mask Tests
# =============================================================================


class TestLabelMask:
    """Tests for label mask correctness."""

    def test_label_mask_is_boolean(self, binding_collator, simple_examples):
        """Label mask should be boolean tensor."""
        batch = binding_collator(simple_examples)
        assert batch["label_mask"].dtype == torch.bool

    def test_label_mask_identifies_residues(self, binding_collator, simple_examples):
        """Label mask should correctly identify residue positions."""
        batch = binding_collator(simple_examples)

        for i, example in enumerate(simple_examples):
            mask = batch["label_mask"][i]
            expected_count = len(example["sequence"])
            actual_count = mask.sum().item()

            assert (
                actual_count == expected_count
            ), f"Example {i}: mask identifies {actual_count} positions, expected {expected_count}"
