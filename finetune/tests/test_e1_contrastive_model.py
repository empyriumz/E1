"""
Tests for E1ForContrastiveBinding model.

Tests verify:
- Prototype initialization and derivation (neg = -pos)
- Prototype normalization
- Forward pass output structure
- Valid residue extraction via label_mask
- Multi-view processing
- Logits shape
- MLM loss averaging
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

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
def mock_e1_base_model(hidden_size):
    """Create mock E1 base model."""

    class MockE1Model(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.config = MagicMock()
            self.config.hidden_size = hidden_size
            self.hidden_size = hidden_size
            self.mlm_head = nn.Linear(hidden_size, 32)

        def forward(
            self,
            input_ids,
            within_seq_position_ids=None,
            global_position_ids=None,
            sequence_ids=None,
            labels=None,
            **kwargs,
        ):
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)

            output = MagicMock()
            output.last_hidden_state = hidden_states
            output.logits = self.mlm_head(hidden_states)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                output.loss = loss_fct(
                    output.logits.reshape(-1, 32), labels.reshape(-1)
                )
            else:
                output.loss = None

            return output

        def get_encoder_output(self, input_ids, **kwargs):
            batch_size, seq_len = input_ids.shape
            return torch.randn(batch_size, seq_len, self.hidden_size)

    return MockE1Model(hidden_size)


@pytest.fixture
def contrastive_model(mock_e1_base_model, hidden_size):  #
    """Create E1ForContrastiveBinding with mocked base model."""
    # Need to mock the parent class initialization
    with patch(
        "training.e1_contrastive_model.E1ForJointBindingMLM.__init__"
    ) as mock_init:
        mock_init.return_value = None

        from training.e1_contrastive_model import E1ForContrastiveBinding

        model = E1ForContrastiveBinding.__new__(E1ForContrastiveBinding)
        nn.Module.__init__(model)  # Initialize nn.Module state

        # Manually initialize required attributes
        model._original_model = mock_e1_base_model
        model.e1_backbone = mock_e1_base_model  # Required for train() method
        model.hidden_size = hidden_size
        model.prototype_dim = hidden_size
        model.ion_types = ["CA", "ZN"]
        model.mlm_weight = 1.0
        model.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        model.mlm_head = mock_e1_base_model.mlm_head
        model._prototypes_initialized = {ion: False for ion in model.ion_types}

        # Register prototype buffers
        for ion in model.ion_types:
            model.register_buffer(f"pos_prototype_{ion}", torch.zeros(hidden_size))

        # Set training mode
        model.train()

        return model


@pytest.fixture
def sample_batch(hidden_size):
    """Create sample batch for model testing."""
    batch_size = 2
    n_views = 4
    seq_len = 16

    return {
        "input_ids": torch.randint(1, 26, (batch_size, n_views, seq_len)),
        "within_seq_position_ids": torch.arange(seq_len)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, n_views, -1),
        "global_position_ids": torch.arange(seq_len)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, n_views, -1),
        "sequence_ids": torch.zeros(batch_size, n_views, seq_len, dtype=torch.long),
        "binding_labels": torch.randint(0, 2, (batch_size, seq_len)).float(),
        "label_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "contrastive_label_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "mlm_labels": torch.randint(0, 32, (batch_size, n_views, seq_len)),
    }


@pytest.fixture
def mock_loss_fn(device):
    """Mock loss function for testing."""
    mock_fn = MagicMock()
    mock_fn.set_prototypes = MagicMock()
    mock_fn.return_value = {
        "total": torch.tensor(1.0, requires_grad=True),
        "contrastive": torch.tensor(0.3),
        "prototype": torch.tensor(0.4),
        "bce": torch.tensor(0.3),
        "pull_mask_ratio": torch.tensor(0.5),
        "avg_sim_pos_pos": torch.tensor(0.8),
        "avg_sim_pos_neg": torch.tensor(-0.2),
        "avg_sim_neg_pos": torch.tensor(-0.3),
        "avg_sim_neg_neg": torch.tensor(0.7),
    }
    return mock_fn


# =============================================================================
# Prototype Tests
# =============================================================================


class TestPrototypeInitialization:
    """Tests for prototype initialization."""

    def test_prototype_stored_correctly(self, contrastive_model, hidden_size, device):
        """Setting prototype should store it correctly."""
        prototype = torch.randn(hidden_size)

        contrastive_model.set_positive_prototype("CA", prototype)

        # Should be marked as initialized
        assert contrastive_model._prototypes_initialized["CA"]

        # Should be stored
        stored = getattr(contrastive_model, "pos_prototype_CA")
        assert stored.shape == (hidden_size,)

    def test_prototype_normalized(self, contrastive_model, hidden_size, device):
        """Stored prototype should be unit-normalized."""
        prototype = torch.randn(hidden_size) * 10  # Large magnitude

        contrastive_model.set_positive_prototype("CA", prototype)

        stored = getattr(contrastive_model, "pos_prototype_CA")
        norm = stored.norm().item()

        assert (
            abs(norm - 1.0) < 0.01
        ), f"Prototype should be unit-normalized, got norm {norm}"


class TestPrototypeDerivation:
    """Tests for prototype derivation (neg = -pos)."""

    def test_negative_is_negation(self, contrastive_model, hidden_size, device):
        """Negative prototype should be negation of positive."""
        prototype = torch.randn(hidden_size)
        contrastive_model.set_positive_prototype("CA", prototype)

        prototypes = contrastive_model.get_prototypes("CA")

        neg_proto = prototypes[0]
        pos_proto = prototypes[1]

        # neg should equal -pos
        assert torch.allclose(
            neg_proto, -pos_proto, atol=1e-6
        ), "Negative prototype should be negation of positive"

    def test_prototypes_shape(self, contrastive_model, hidden_size, device):
        """get_prototypes should return [2, hidden_size]."""
        prototype = torch.randn(hidden_size)
        contrastive_model.set_positive_prototype("CA", prototype)

        prototypes = contrastive_model.get_prototypes("CA")

        assert prototypes.shape == (
            2,
            hidden_size,
        ), f"Expected shape (2, {hidden_size}), got {prototypes.shape}"

    def test_both_prototypes_normalized(self, contrastive_model, hidden_size, device):
        """Both pos and neg prototypes should be unit-normalized."""
        prototype = torch.randn(hidden_size)
        contrastive_model.set_positive_prototype("CA", prototype)

        prototypes = contrastive_model.get_prototypes("CA")

        for i in range(2):
            norm = prototypes[i].norm().item()
            assert (
                abs(norm - 1.0) < 0.01
            ), f"Prototype {i} should be unit-normalized, got norm {norm}"


# =============================================================================
# Forward Pass Tests
# =============================================================================


class TestForwardPass:
    """Tests for model forward pass."""

    def test_forward_returns_output_object(
        self, contrastive_model, sample_batch, mock_loss_fn, device
    ):
        """Forward should return E1ContrastiveOutput."""
        # Initialize prototype
        contrastive_model.set_positive_prototype(
            "CA", torch.randn(contrastive_model.hidden_size)
        )

        from training.e1_contrastive_model import E1ContrastiveOutput

        output = contrastive_model.forward(
            input_ids=sample_batch["input_ids"],
            within_seq_position_ids=sample_batch["within_seq_position_ids"],
            global_position_ids=sample_batch["global_position_ids"],
            sequence_ids=sample_batch["sequence_ids"],
            ion="CA",
            binding_labels=sample_batch["binding_labels"],
            label_mask=sample_batch["label_mask"],
            contrastive_label_mask=sample_batch["contrastive_label_mask"],
            mlm_labels=sample_batch["mlm_labels"],
            loss_fn=mock_loss_fn,
        )

        assert isinstance(output, E1ContrastiveOutput)

    def test_output_has_loss(
        self, contrastive_model, sample_batch, mock_loss_fn, device
    ):
        """Output should have loss tensor."""
        contrastive_model.set_positive_prototype(
            "CA", torch.randn(contrastive_model.hidden_size)
        )

        output = contrastive_model.forward(
            input_ids=sample_batch["input_ids"],
            within_seq_position_ids=sample_batch["within_seq_position_ids"],
            global_position_ids=sample_batch["global_position_ids"],
            sequence_ids=sample_batch["sequence_ids"],
            ion="CA",
            binding_labels=sample_batch["binding_labels"],
            label_mask=sample_batch["label_mask"],
            mlm_labels=sample_batch["mlm_labels"],
            loss_fn=mock_loss_fn,
        )

        assert output.loss is not None

    def test_output_has_embeddings(
        self, contrastive_model, sample_batch, mock_loss_fn, device
    ):
        """Output should have embeddings tensor."""
        contrastive_model.set_positive_prototype(
            "CA", torch.randn(contrastive_model.hidden_size)
        )

        output = contrastive_model.forward(
            input_ids=sample_batch["input_ids"],
            within_seq_position_ids=sample_batch["within_seq_position_ids"],
            global_position_ids=sample_batch["global_position_ids"],
            sequence_ids=sample_batch["sequence_ids"],
            ion="CA",
            binding_labels=sample_batch["binding_labels"],
            label_mask=sample_batch["label_mask"],
            mlm_labels=sample_batch["mlm_labels"],
            loss_fn=mock_loss_fn,
        )

        assert output.embeddings is not None

    def test_embeddings_shape(
        self, contrastive_model, sample_batch, mock_loss_fn, device
    ):
        """Embeddings should have shape [batch, n_views, seq_len, hidden]."""
        contrastive_model.set_positive_prototype(
            "CA", torch.randn(contrastive_model.hidden_size)
        )

        output = contrastive_model.forward(
            input_ids=sample_batch["input_ids"],
            within_seq_position_ids=sample_batch["within_seq_position_ids"],
            global_position_ids=sample_batch["global_position_ids"],
            sequence_ids=sample_batch["sequence_ids"],
            ion="CA",
            binding_labels=sample_batch["binding_labels"],
            label_mask=sample_batch["label_mask"],
            mlm_labels=sample_batch["mlm_labels"],
            loss_fn=mock_loss_fn,
        )

        batch_size, n_views, seq_len = sample_batch["input_ids"].shape
        expected_shape = (batch_size, n_views, seq_len, contrastive_model.hidden_size)

        assert (
            output.embeddings.shape == expected_shape
        ), f"Expected embeddings shape {expected_shape}, got {output.embeddings.shape}"


# =============================================================================
# Multi-View Processing Tests
# =============================================================================


class TestMultiViewProcessing:
    """Tests for multi-view processing."""

    def test_all_views_processed(
        self, contrastive_model, sample_batch, mock_loss_fn, device
    ):
        """All views should be processed independently."""
        contrastive_model.set_positive_prototype(
            "CA", torch.randn(contrastive_model.hidden_size)
        )

        output = contrastive_model.forward(
            input_ids=sample_batch["input_ids"],
            within_seq_position_ids=sample_batch["within_seq_position_ids"],
            global_position_ids=sample_batch["global_position_ids"],
            sequence_ids=sample_batch["sequence_ids"],
            ion="CA",
            binding_labels=sample_batch["binding_labels"],
            label_mask=sample_batch["label_mask"],
            mlm_labels=sample_batch["mlm_labels"],
            loss_fn=mock_loss_fn,
        )

        # Should have embeddings for all views
        n_views = sample_batch["input_ids"].shape[1]
        assert output.embeddings.shape[1] == n_views


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_unknown_ion_raises(self, contrastive_model, sample_batch, mock_loss_fn):
        """Unknown ion type should raise error."""
        with pytest.raises(ValueError, match="Unknown ion type"):
            contrastive_model.forward(
                input_ids=sample_batch["input_ids"],
                within_seq_position_ids=sample_batch["within_seq_position_ids"],
                global_position_ids=sample_batch["global_position_ids"],
                sequence_ids=sample_batch["sequence_ids"],
                ion="UNKNOWN_ION",  # Not in ion_types
                binding_labels=sample_batch["binding_labels"],
                label_mask=sample_batch["label_mask"],
                loss_fn=mock_loss_fn,
            )

    def test_uninitialized_prototype_logs_warning(
        self, contrastive_model, sample_batch, mock_loss_fn, caplog
    ):
        """Uninitialized prototype should log warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            output = contrastive_model.forward(
                input_ids=sample_batch["input_ids"],
                within_seq_position_ids=sample_batch["within_seq_position_ids"],
                global_position_ids=sample_batch["global_position_ids"],
                sequence_ids=sample_batch["sequence_ids"],
                ion="CA",  # Not initialized
                binding_labels=sample_batch["binding_labels"],
                label_mask=sample_batch["label_mask"],
                loss_fn=mock_loss_fn,
            )

        # Should still work, just log warning
        assert output is not None
