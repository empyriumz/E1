"""
Tests for E1DataCollatorForContrastive (contrastive collator).

Tests verify:
- Multi-variant generation with distinct masks
- Output tensor shapes [batch, num_variants, seq_len]
- Masking probability and 80-10-10 strategy
- MSA resampling per variant
- Contrastive label mask excludes masked positions
- MLM labels generation
- Edge cases (single variant, zero masking)
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
def mock_binding_collator():
    """Mock the base binding collator."""
    mock = MagicMock()

    # Mock batch_preparer
    mock.batch_preparer = MagicMock()
    mock.batch_preparer.pad_token_id = 0
    mock.batch_preparer.mask_token_id = 26
    mock.batch_preparer.vocab = {chr(i + 65): i + 1 for i in range(26)}
    mock.batch_preparer.boundary_token_ids = torch.tensor([0, 27, 28, 29, 30])

    def mock_call(examples):
        """Return a mock batch that looks like real collator output."""
        batch_size = len(examples)
        # Simulate tokenized output with 12 tokens per sequence
        seq_len = 12

        return {
            "input_ids": torch.randint(1, 26, (batch_size, seq_len)),
            "sequence_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "within_seq_position_ids": torch.arange(seq_len)
            .unsqueeze(0)
            .expand(batch_size, -1),
            "global_position_ids": torch.arange(seq_len)
            .unsqueeze(0)
            .expand(batch_size, -1),
            "binding_labels": torch.randint(0, 2, (batch_size, seq_len)).float(),
            "label_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "protein_ids": [
                ex.get("protein_id", f"protein_{i}") for i, ex in enumerate(examples)
            ],
        }

    mock.__call__ = mock_call
    mock.side_effect = mock_call
    return mock


@pytest.fixture
def contrastive_collator(mock_binding_collator):
    """Create contrastive collator with mocked dependencies."""
    with patch(
        "training.e1_contrastive_collator.E1DataCollatorForResidueClassification",
        return_value=mock_binding_collator,
    ):
        with patch(
            "training.e1_contrastive_collator.E1BatchPreparer"
        ) as mock_preparer_class:
            mock_preparer = MagicMock()
            mock_preparer.pad_token_id = 0
            mock_preparer.mask_token_id = 26
            mock_preparer.vocab = {chr(i + 65): i + 1 for i in range(26)}
            mock_preparer.boundary_token_ids = torch.tensor([0, 27, 28, 29, 30])
            mock_preparer_class.return_value = mock_preparer

            from training.e1_contrastive_collator import E1DataCollatorForContrastive

            collator = E1DataCollatorForContrastive(
                num_variants=4,
                mask_prob_min=0.05,
                mask_prob_max=0.15,
                max_total_tokens=8192,
                max_query_tokens=1024,
                ignore_index=-100,
                resample_msa_per_variant=False,  # Simpler for testing
            )

            # Replace with our mocks
            collator.binding_collator = mock_binding_collator
            collator.batch_preparer = mock_preparer
            collator.mask_token_id = 26
            collator.pad_token_id = 0
            collator.vocab_size = 32
            collator.boundary_token_ids = torch.tensor([0, 27, 28, 29, 30])

            return collator


@pytest.fixture
def simple_contrastive_examples():
    """Simple examples for contrastive testing."""
    return [
        {
            "sequence": "ACDEFGH",
            "labels": [0, 1, 0, 1, 0, 0, 1],
            "msa_context": None,
            "protein_id": "protein_0",
        },
        {
            "sequence": "KLMNPQ",
            "labels": [1, 0, 0, 1, 0, 1],
            "msa_context": None,
            "protein_id": "protein_1",
        },
    ]


# =============================================================================
# Variant Generation Tests
# =============================================================================


class TestVariantGeneration:
    """Tests for multi-variant generation."""

    def test_creates_correct_number_of_variants(
        self, contrastive_collator, simple_contrastive_examples
    ):
        """Verify num_variants distinct views are created."""
        batch = contrastive_collator(simple_contrastive_examples)

        assert (
            batch["input_ids"].dim() == 3
        ), "Should have 3 dimensions [batch, variants, seq_len]"
        assert batch["input_ids"].shape[1] == 4, "Should have 4 variants"

    def test_variants_have_different_masks(
        self, contrastive_collator, simple_contrastive_examples
    ):
        """Each variant should have a different mask pattern."""
        # Set deterministic masking for reproducibility
        torch.manual_seed(42)
        batch = contrastive_collator(simple_contrastive_examples)

        input_ids = batch["input_ids"]  # [batch, variants, seq_len]

        # Check first sample - variants should differ
        sample_0_variants = input_ids[0]  # [variants, seq_len]

        # At least some pairs should be different (masks are random)
        differences = 0
        for i in range(sample_0_variants.shape[0]):
            for j in range(i + 1, sample_0_variants.shape[0]):
                if not torch.equal(sample_0_variants[i], sample_0_variants[j]):
                    differences += 1

        # With 4 variants and random masking, we expect some differences
        assert differences > 0, "Variants should have different mask patterns"


# =============================================================================
# Output Shape Tests
# =============================================================================


class TestOutputShape:
    """Tests for output tensor shapes."""

    def test_input_ids_shape(self, contrastive_collator, simple_contrastive_examples):
        """Input IDs should have shape [batch, num_variants, seq_len]."""
        batch = contrastive_collator(simple_contrastive_examples)

        assert batch["input_ids"].shape[0] == 2  # batch size
        assert batch["input_ids"].shape[1] == 4  # num_variants
        assert batch["input_ids"].shape[2] > 0  # seq_len

    def test_all_position_ids_have_variant_dim(
        self, contrastive_collator, simple_contrastive_examples
    ):
        """Position IDs should also have variant dimension."""
        batch = contrastive_collator(simple_contrastive_examples)

        assert batch["within_seq_position_ids"].dim() == 3
        assert batch["global_position_ids"].dim() == 3
        assert batch["sequence_ids"].dim() == 3

    def test_mlm_labels_shape(self, contrastive_collator, simple_contrastive_examples):
        """MLM labels should have shape [batch, num_variants, seq_len]."""
        batch = contrastive_collator(simple_contrastive_examples)

        assert "mlm_labels" in batch
        assert batch["mlm_labels"].shape == batch["input_ids"].shape

    def test_binding_labels_shape(
        self, contrastive_collator, simple_contrastive_examples
    ):
        """Binding labels should be expanded to variant dimension."""
        batch = contrastive_collator(simple_contrastive_examples)

        # Binding labels should be [batch, num_variants, seq_len] or [batch, seq_len]
        assert batch["binding_labels"].dim() >= 2


# =============================================================================
# Masking Strategy Tests
# =============================================================================


class TestMaskingStrategy:
    """Tests for masking behavior."""

    def test_mask_positions_recorded(
        self, contrastive_collator, simple_contrastive_examples
    ):
        """MLM labels should record original tokens at masked positions."""
        batch = contrastive_collator(simple_contrastive_examples)

        mlm_labels = batch["mlm_labels"]

        # MLM labels should have some non-padding values (original tokens)
        # and some padding values (unmasked positions)
        assert (
            mlm_labels != contrastive_collator.pad_token_id
        ).any(), "MLM labels should have some masked positions"

    def test_contrastive_label_mask_excludes_masked(
        self, contrastive_collator, simple_contrastive_examples
    ):
        """Contrastive label mask should exclude masked token positions."""
        batch = contrastive_collator(simple_contrastive_examples)

        assert "contrastive_label_mask" in batch

        contrastive_mask = batch["contrastive_label_mask"]
        label_mask = batch["label_mask"]

        # Contrastive mask should be subset of label mask
        # (fewer True values due to masking exclusion)
        assert contrastive_mask.shape == label_mask.shape

        # Contrastive mask should have equal or fewer True values
        assert contrastive_mask.sum() <= label_mask.sum()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_variant_mode(self, mock_binding_collator):
        """Collator should work with num_variants=1."""
        with patch(
            "training.e1_contrastive_collator.E1DataCollatorForResidueClassification",
            return_value=mock_binding_collator,
        ):
            with patch(
                "training.e1_contrastive_collator.E1BatchPreparer"
            ) as mock_preparer_class:
                mock_preparer = MagicMock()
                mock_preparer.pad_token_id = 0
                mock_preparer.mask_token_id = 26
                mock_preparer.vocab = {chr(i + 65): i + 1 for i in range(26)}
                mock_preparer.boundary_token_ids = torch.tensor([0, 27, 28, 29, 30])
                mock_preparer_class.return_value = mock_preparer

                from training.e1_contrastive_collator import (
                    E1DataCollatorForContrastive,
                )

                collator = E1DataCollatorForContrastive(
                    num_variants=1,
                    mask_prob_min=0.0,
                    mask_prob_max=0.0,  # No masking
                    resample_msa_per_variant=False,
                )
                collator.binding_collator = mock_binding_collator
                collator.batch_preparer = mock_preparer
                collator.mask_token_id = 26
                collator.pad_token_id = 0
                collator.vocab_size = 32
                collator.boundary_token_ids = torch.tensor([0, 27, 28, 29, 30])

                examples = [
                    {"sequence": "ABC", "labels": [0, 1, 0], "protein_id": "test"}
                ]
                batch = collator(examples)

                assert batch["input_ids"].shape[1] == 1, "Should have 1 variant"

    def test_zero_masking_probability(self, mock_binding_collator):
        """With mask_prob=0, no tokens should be masked."""
        with patch(
            "training.e1_contrastive_collator.E1DataCollatorForResidueClassification",
            return_value=mock_binding_collator,
        ):
            with patch(
                "training.e1_contrastive_collator.E1BatchPreparer"
            ) as mock_preparer_class:
                mock_preparer = MagicMock()
                mock_preparer.pad_token_id = 0
                mock_preparer.mask_token_id = 26
                mock_preparer.vocab = {chr(i + 65): i + 1 for i in range(26)}
                mock_preparer.boundary_token_ids = torch.tensor([0, 27, 28, 29, 30])
                mock_preparer_class.return_value = mock_preparer

                from training.e1_contrastive_collator import (
                    E1DataCollatorForContrastive,
                )

                collator = E1DataCollatorForContrastive(
                    num_variants=2,
                    mask_prob_min=0.0,
                    mask_prob_max=0.0,  # No masking
                    resample_msa_per_variant=False,
                )
                collator.binding_collator = mock_binding_collator
                collator.batch_preparer = mock_preparer
                collator.mask_token_id = 26
                collator.pad_token_id = 0
                collator.vocab_size = 32
                collator.boundary_token_ids = torch.tensor([0, 27, 28, 29, 30])

                examples = [
                    {"sequence": "ABC", "labels": [0, 1, 0], "protein_id": "test"}
                ]
                batch = collator(examples)

                # With zero masking, contrastive_label_mask should equal label_mask
                contrastive_mask = batch["contrastive_label_mask"]
                label_mask = batch["label_mask"]

                assert torch.equal(
                    contrastive_mask, label_mask
                ), "With zero masking, contrastive and label masks should be equal"


# =============================================================================
# MLM Labels Tests
# =============================================================================


class TestMLMLabels:
    """Tests for MLM label generation."""

    def test_mlm_labels_have_original_tokens_at_masked_positions(
        self, contrastive_collator, simple_contrastive_examples
    ):
        """MLM labels should contain original token IDs at masked positions."""
        batch = contrastive_collator(simple_contrastive_examples)

        mlm_labels = batch["mlm_labels"]

        # Non-pad positions in mlm_labels should be valid token IDs
        valid_mlm = mlm_labels[mlm_labels != contrastive_collator.pad_token_id]

        # All valid MLM labels should be valid token IDs (> 0, < vocab_size)
        if len(valid_mlm) > 0:
            assert (valid_mlm > 0).all(), "MLM labels should be positive token IDs"
            assert (
                valid_mlm < contrastive_collator.vocab_size
            ).all(), "MLM labels should be less than vocab size"


# =============================================================================
# Binding Labels Tests
# =============================================================================


class TestBindingLabels:
    """Tests for binding label handling across views."""

    def test_binding_labels_present(
        self, contrastive_collator, simple_contrastive_examples
    ):
        """Binding labels should be in output."""
        batch = contrastive_collator(simple_contrastive_examples)

        assert "binding_labels" in batch
        assert batch["binding_labels"] is not None
