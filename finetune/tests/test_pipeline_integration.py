"""
Integration tests for E1 contrastive training pipeline.

Tests the full data flow from:
- Raw sequences → Collator → Model forward
- Validates input/output formats at each stage
- Tests combined behavior of components
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
def device():
    """CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def hidden_size():
    """Standard hidden size."""
    return 64


@pytest.fixture
def mock_batch_preparer():
    """Mock E1BatchPreparer for integration testing."""
    mock = MagicMock()
    mock.pad_token_id = 0
    mock.mask_token_id = 26
    mock.vocab = {chr(i + 65): i + 1 for i in range(26)}
    mock.vocab.update({"<bos>": 27, "<eos>": 28, "1": 29, "2": 30, "<pad>": 0, "?": 26})
    mock.boundary_token_ids = torch.tensor([0, 27, 28, 29, 30])

    def get_batch_kwargs(sequences, device=torch.device("cpu")):
        batch_size = len(sequences)
        max_len = 20  # Fixed for simplicity

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        sequence_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        within_pos = torch.zeros(batch_size, max_len, dtype=torch.long)
        global_pos = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, seq in enumerate(sequences):
            parts = seq.split(",")
            query = parts[-1]
            tokens = (
                [27, 29]
                + [ord(c) - ord("A") + 1 for c in query[: max_len - 4]]
                + [30, 28]
            )
            seq_len = len(tokens)
            input_ids[i, :seq_len] = torch.tensor(tokens)
            sequence_ids[i, :seq_len] = len(parts) - 1
            within_pos[i, :seq_len] = torch.arange(seq_len)
            global_pos[i, :seq_len] = torch.arange(seq_len)

        return {
            "input_ids": input_ids.to(device),
            "sequence_ids": sequence_ids.to(device),
            "within_seq_position_ids": within_pos.to(device),
            "global_position_ids": global_pos.to(device),
        }

    mock.get_batch_kwargs = get_batch_kwargs
    return mock


@pytest.fixture
def integration_examples():
    """Sample examples for integration testing."""
    return [
        {
            "sequence": "ACDEFGHIK",
            "labels": [0, 1, 0, 0, 1, 0, 0, 0, 1],
            "msa_context": None,
            "protein_id": "protein_0",
        },
        {
            "sequence": "LMNPQRST",
            "labels": [1, 0, 0, 1, 0, 1, 0, 0],
            "msa_context": None,
            "protein_id": "protein_1",
        },
    ]


# =============================================================================
# Data Flow Tests
# =============================================================================


class TestDataFlow:
    """Tests for end-to-end data flow."""

    def test_collator_output_compatible_with_model_input(
        self, mock_batch_preparer, integration_examples
    ):
        """Collator output should be compatible with model forward."""
        with patch(
            "training.e1_contrastive_collator.E1DataCollatorForResidueClassification"
        ) as mock_binding:
            with patch(
                "training.e1_contrastive_collator.E1BatchPreparer",
                return_value=mock_batch_preparer,
            ):
                # Setup mock binding collator
                def mock_binding_call(examples):
                    batch_size = len(examples)
                    seq_len = 15
                    return {
                        "input_ids": torch.randint(1, 26, (batch_size, seq_len)),
                        "sequence_ids": torch.zeros(
                            batch_size, seq_len, dtype=torch.long
                        ),
                        "within_seq_position_ids": torch.arange(seq_len)
                        .unsqueeze(0)
                        .expand(batch_size, -1),
                        "global_position_ids": torch.arange(seq_len)
                        .unsqueeze(0)
                        .expand(batch_size, -1),
                        "binding_labels": torch.randint(
                            0, 2, (batch_size, seq_len)
                        ).float(),
                        "label_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
                        "protein_ids": [ex.get("protein_id") for ex in examples],
                    }

                mock_binding_instance = MagicMock()
                mock_binding_instance.side_effect = mock_binding_call
                mock_binding.return_value = mock_binding_instance

                from training.e1_contrastive_collator import (
                    E1DataCollatorForContrastive,
                )

                collator = E1DataCollatorForContrastive(
                    num_variants=4,
                    mask_prob_min=0.05,
                    mask_prob_max=0.15,
                    resample_msa_per_variant=False,
                )
                collator.binding_collator = mock_binding_instance
                collator.batch_preparer = mock_batch_preparer
                collator.mask_token_id = 26
                collator.pad_token_id = 0
                collator.vocab_size = 32
                collator.boundary_token_ids = mock_batch_preparer.boundary_token_ids

                # Get collator output
                batch = collator(integration_examples)

                # Verify output has all required keys for model
                required_keys = [
                    "input_ids",
                    "within_seq_position_ids",
                    "global_position_ids",
                    "sequence_ids",
                    "mlm_labels",
                    "binding_labels",
                    "label_mask",
                    "contrastive_label_mask",
                ]

                for key in required_keys:
                    assert key in batch, f"Missing key: {key}"

                # Verify shapes are compatible
                batch_size = len(integration_examples)
                num_variants = 4

                assert batch["input_ids"].shape[0] == batch_size
                assert batch["input_ids"].shape[1] == num_variants
                assert batch["input_ids"].dim() == 3  # [batch, variants, seq_len]


class TestContrastiveLabelMaskFlow:
    """Tests for contrastive label mask propagation through pipeline."""

    def test_contrastive_mask_excludes_masked_tokens(
        self, mock_batch_preparer, integration_examples
    ):
        """Contrastive label mask should exclude positions that were masked."""
        with patch(
            "training.e1_contrastive_collator.E1DataCollatorForResidueClassification"
        ) as mock_binding:
            with patch(
                "training.e1_contrastive_collator.E1BatchPreparer",
                return_value=mock_batch_preparer,
            ):

                def mock_binding_call(examples):
                    batch_size = len(examples)
                    seq_len = 15
                    return {
                        "input_ids": torch.randint(1, 26, (batch_size, seq_len)),
                        "sequence_ids": torch.zeros(
                            batch_size, seq_len, dtype=torch.long
                        ),
                        "within_seq_position_ids": torch.arange(seq_len)
                        .unsqueeze(0)
                        .expand(batch_size, -1),
                        "global_position_ids": torch.arange(seq_len)
                        .unsqueeze(0)
                        .expand(batch_size, -1),
                        "binding_labels": torch.randint(
                            0, 2, (batch_size, seq_len)
                        ).float(),
                        "label_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
                        "protein_ids": [ex.get("protein_id") for ex in examples],
                    }

                mock_binding_instance = MagicMock()
                mock_binding_instance.side_effect = mock_binding_call
                mock_binding.return_value = mock_binding_instance

                from training.e1_contrastive_collator import (
                    E1DataCollatorForContrastive,
                )

                # Use high masking to ensure some tokens are masked
                collator = E1DataCollatorForContrastive(
                    num_variants=2,
                    mask_prob_min=0.3,  # High masking
                    mask_prob_max=0.5,
                    resample_msa_per_variant=False,
                )
                collator.binding_collator = mock_binding_instance
                collator.batch_preparer = mock_batch_preparer
                collator.mask_token_id = 26
                collator.pad_token_id = 0
                collator.vocab_size = 32
                collator.boundary_token_ids = mock_batch_preparer.boundary_token_ids

                torch.manual_seed(42)  # For reproducibility
                batch = collator(integration_examples)

                label_mask = batch["label_mask"]
                contrastive_mask = batch["contrastive_label_mask"]

                # Contrastive mask should be subset of label mask
                assert (
                    contrastive_mask.sum() <= label_mask.sum()
                ), "Contrastive mask should have fewer or equal True values"


# =============================================================================
# Input Format Tests
# =============================================================================


class TestInputFormats:
    """Tests validating input formats at each pipeline stage."""

    def test_loss_function_input_format(self, hidden_size, device):
        """Verify input format to loss function is [N, n_views, hidden]."""
        from loss_func_contrastive import PrototypeBCELoss

        # Create prototypes
        pos_proto = torch.randn(hidden_size)
        pos_proto = pos_proto / pos_proto.norm()
        prototypes = torch.stack([-pos_proto, pos_proto], dim=0)

        # Create loss function
        loss_fn = PrototypeBCELoss(
            temperature=0.07,
            prototype_weight=1.0,
            unsupervised_weight=1.0,
            bce_weight=1.0,
            device=device,
        )
        loss_fn.set_prototypes(prototypes)

        # Input format: [total_residues, n_views, hidden]
        total_residues = 32
        n_views = 4
        features = torch.randn(total_residues, n_views, hidden_size, requires_grad=True)
        labels = torch.randint(0, 2, (total_residues,))

        # Should work without error
        result = loss_fn(features, labels)

        assert "total" in result
        assert result["total"].requires_grad

    def test_prototype_scoring_input_format(self, hidden_size, device):
        """Verify input format to prototype scoring."""
        from training.prototype_scoring import compute_prototype_distance_scores

        # Create prototypes
        pos_proto = torch.randn(hidden_size)
        pos_proto = pos_proto / pos_proto.norm()
        prototypes = torch.stack([-pos_proto, pos_proto], dim=0)

        # Test 2D input: [batch, hidden]
        embeddings_2d = torch.randn(16, hidden_size)
        scores_2d = compute_prototype_distance_scores(embeddings_2d, prototypes)
        assert scores_2d.shape == (
            16,
        ), f"2D input should give [batch] output, got {scores_2d.shape}"

        # Test 3D input: [batch, n_views, hidden]
        embeddings_3d = torch.randn(16, 4, hidden_size)
        scores_3d = compute_prototype_distance_scores(embeddings_3d, prototypes)
        assert scores_3d.shape == (
            16,
        ), f"3D input should give [batch] output (averaged), got {scores_3d.shape}"


class TestValidResidueExtraction:
    """Tests for valid residue extraction logic."""

    def test_label_mask_filtering(self, device):
        """Verify that label_mask correctly filters valid residues."""
        batch_size = 2
        seq_len = 10
        hidden_size = 32
        n_views = 4

        # Create mock hidden states
        hidden_states = torch.randn(batch_size, n_views, seq_len, hidden_size)

        # Create label mask with some invalid positions
        label_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        label_mask[0, 2:7] = True  # 5 valid positions for sample 0
        label_mask[1, 3:8] = True  # 5 valid positions for sample 1

        # Create binding labels
        binding_labels = torch.randint(0, 2, (batch_size, seq_len)).float()

        # Extract valid residues (simulating model logic)
        all_features = []
        all_labels = []

        for b in range(batch_size):
            valid_pos = label_mask[b]
            if valid_pos.sum() > 0:
                # [n_views, valid_count, hidden] -> [valid_count, n_views, hidden]
                sample_emb = hidden_states[b, :, valid_pos, :].transpose(0, 1)
                sample_labels = binding_labels[b, valid_pos]
                all_features.append(sample_emb)
                all_labels.append(sample_labels)

        features = torch.cat(all_features, dim=0)  # [total_valid, n_views, hidden]
        labels = torch.cat(all_labels, dim=0)  # [total_valid]

        # Verify shapes
        expected_valid = label_mask.sum().item()  # 5 + 5 = 10
        assert features.shape == (
            expected_valid,
            n_views,
            hidden_size,
        ), f"Expected shape ({expected_valid}, {n_views}, {hidden_size}), got {features.shape}"
        assert labels.shape == (expected_valid,)


# =============================================================================
# MSA Sampling Tests
# =============================================================================


class TestMSASamplingBehavior:
    """Tests for MSA sampling behavior in pipeline."""

    def test_different_variants_can_have_different_msa(self):
        """When resample_msa_per_variant=True, variants should have different MSA."""
        # This is a conceptual test - in practice, MSA resampling requires actual MSA files
        # Here we just verify the collator accepts the parameter

        with patch(
            "training.e1_contrastive_collator.E1DataCollatorForResidueClassification"
        ):
            with patch("training.e1_contrastive_collator.E1BatchPreparer"):
                from training.e1_contrastive_collator import (
                    E1DataCollatorForContrastive,
                )

                # Should not raise error
                collator = E1DataCollatorForContrastive(
                    num_variants=4,
                    mask_prob_min=0.05,
                    mask_prob_max=0.15,
                    resample_msa_per_variant=True,
                    msa_sampling_config={
                        "max_num_samples": 32,
                        "max_token_length": 4096,
                    },
                )

                assert collator.resample_msa_per_variant is True


# =============================================================================
# Edge Case Integration Tests
# =============================================================================


class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_single_sample_batch(self, mock_batch_preparer):
        """Pipeline should work with single sample batch."""
        with patch(
            "training.e1_contrastive_collator.E1DataCollatorForResidueClassification"
        ) as mock_binding:
            with patch(
                "training.e1_contrastive_collator.E1BatchPreparer",
                return_value=mock_batch_preparer,
            ):

                def mock_binding_call(examples):
                    batch_size = len(examples)
                    seq_len = 10
                    return {
                        "input_ids": torch.randint(1, 26, (batch_size, seq_len)),
                        "sequence_ids": torch.zeros(
                            batch_size, seq_len, dtype=torch.long
                        ),
                        "within_seq_position_ids": torch.arange(seq_len)
                        .unsqueeze(0)
                        .expand(batch_size, -1),
                        "global_position_ids": torch.arange(seq_len)
                        .unsqueeze(0)
                        .expand(batch_size, -1),
                        "binding_labels": torch.randint(
                            0, 2, (batch_size, seq_len)
                        ).float(),
                        "label_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
                        "protein_ids": [ex.get("protein_id") for ex in examples],
                    }

                mock_binding_instance = MagicMock()
                mock_binding_instance.side_effect = mock_binding_call
                mock_binding.return_value = mock_binding_instance

                from training.e1_contrastive_collator import (
                    E1DataCollatorForContrastive,
                )

                collator = E1DataCollatorForContrastive(
                    num_variants=2,
                    mask_prob_min=0.05,
                    mask_prob_max=0.15,
                    resample_msa_per_variant=False,
                )
                collator.binding_collator = mock_binding_instance
                collator.batch_preparer = mock_batch_preparer
                collator.mask_token_id = 26
                collator.pad_token_id = 0
                collator.vocab_size = 32
                collator.boundary_token_ids = mock_batch_preparer.boundary_token_ids

                # Single sample
                single_example = [
                    {
                        "sequence": "ACDEF",
                        "labels": [0, 1, 0, 1, 0],
                        "protein_id": "single_protein",
                    }
                ]

                batch = collator(single_example)

                assert batch["input_ids"].shape[0] == 1  # Batch size = 1

    def test_all_positive_labels(self, hidden_size, device):
        """Loss function should handle all-positive batch."""
        from loss_func_contrastive import PrototypeBCELoss

        pos_proto = torch.randn(hidden_size)
        pos_proto = pos_proto / pos_proto.norm()
        prototypes = torch.stack([-pos_proto, pos_proto], dim=0)

        loss_fn = PrototypeBCELoss(
            temperature=0.07,
            prototype_weight=1.0,
            bce_weight=1.0,
            device=device,
        )
        loss_fn.set_prototypes(prototypes)

        features = torch.randn(16, 4, hidden_size, requires_grad=True)
        labels = torch.ones(16)  # All positive

        result = loss_fn(features, labels)

        assert not torch.isnan(result["total"]), "Should handle all-positive batch"

    def test_all_negative_labels(self, hidden_size, device):
        """Loss function should handle all-negative batch."""
        from loss_func_contrastive import PrototypeBCELoss

        pos_proto = torch.randn(hidden_size)
        pos_proto = pos_proto / pos_proto.norm()
        prototypes = torch.stack([-pos_proto, pos_proto], dim=0)

        loss_fn = PrototypeBCELoss(
            temperature=0.07,
            prototype_weight=1.0,
            bce_weight=1.0,
            device=device,
        )
        loss_fn.set_prototypes(prototypes)

        features = torch.randn(16, 4, hidden_size, requires_grad=True)
        labels = torch.zeros(16)  # All negative

        result = loss_fn(features, labels)

        assert not torch.isnan(result["total"]), "Should handle all-negative batch"
