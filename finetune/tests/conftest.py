"""
Shared pytest fixtures for E1 contrastive pipeline testing.

Provides synthetic test data and mock configurations to enable
testing without GPU or downloading large model weights.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Add finetune and src directories to path for imports
finetune_dir = Path(__file__).parent.parent
if str(finetune_dir) not in sys.path:
    sys.path.insert(0, str(finetune_dir))

src_dir = finetune_dir.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


# =============================================================================
# Device fixture
# =============================================================================


@pytest.fixture
def device():
    """CPU device for testing (no GPU required)."""
    return torch.device("cpu")


# =============================================================================
# Synthetic test data fixtures
# =============================================================================


@pytest.fixture
def sample_sequences():
    """Simple amino acid sequences for testing."""
    return [
        "ACDEFGHIKLMNPQRSTVWY",  # 20 residues (all standard AAs)
        "MKTAYIAKQRQISFVKSH",  # 18 residues
        "GALMFWKGERC",  # 11 residues
    ]


@pytest.fixture
def sample_labels():
    """Binary labels matching sample sequences."""
    return [
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 20 labels
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 18 labels
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 11 labels
    ]


@pytest.fixture
def sample_msa_context():
    """Simple MSA context strings (comma-separated homologs)."""
    return [
        "ACDEFGHIKLMNPQRSTVWY",  # Same as query (single sequence)
        "MKTAYIAKQRQISFVKSH,MKTAYIAKQRQISFVKSH",  # Query with one homolog
        "GALMFWKGERC,GALMFWKGERC,GALMFWKGERC",  # Query with two homologs
    ]


@pytest.fixture
def sample_batch_examples(sample_sequences, sample_labels, sample_msa_context):
    """Complete batch examples for collator testing."""
    return [
        {
            "sequence": sample_sequences[0],
            "labels": sample_labels[0],
            "msa_context": None,  # No MSA context
            "protein_id": "protein_0",
        },
        {
            "sequence": sample_sequences[1],
            "labels": sample_labels[1],
            "msa_context": "MKSAYIAKQRQISFVKSH",  # One homolog
            "protein_id": "protein_1",
        },
    ]


@pytest.fixture
def single_example():
    """Single example for basic tests."""
    return {
        "sequence": "ACDEFGHIK",
        "labels": [0, 1, 0, 0, 1, 0, 0, 0, 1],
        "msa_context": None,
        "protein_id": "test_protein",
    }


# =============================================================================
# Mock E1 model fixtures
# =============================================================================


@pytest.fixture
def mock_hidden_size():
    """Standard hidden size for testing."""
    return 64  # Small for fast tests


@pytest.fixture
def mock_e1_backbone(mock_hidden_size):
    """
    Mock E1 backbone model that returns fake hidden states.

    This avoids loading the real E1 model which requires GPU and large weights.
    """

    class MockE1Model(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.config = MagicMock()
            self.config.hidden_size = hidden_size
            self.hidden_size = hidden_size
            # Mock MLM head
            self.mlm_head = nn.Linear(hidden_size, 32)  # 32 = mock vocab size

        def forward(
            self,
            input_ids,
            within_seq_position_ids=None,
            global_position_ids=None,
            sequence_ids=None,
            labels=None,
            output_hidden_states=False,
            output_attentions=False,
            **kwargs,
        ):
            batch_size, seq_len = input_ids.shape
            # Generate random hidden states
            hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)

            # Create output object
            output = MagicMock()
            output.last_hidden_state = hidden_states
            output.logits = self.mlm_head(hidden_states)

            # Compute MLM loss if labels provided
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                output.loss = loss_fct(output.logits.view(-1, 32), labels.view(-1))
            else:
                output.loss = None

            return output

        def get_encoder_output(
            self,
            input_ids,
            within_seq_position_ids=None,
            global_position_ids=None,
            sequence_ids=None,
            **kwargs,
        ):
            """Return just the hidden states."""
            batch_size, seq_len = input_ids.shape
            return torch.randn(batch_size, seq_len, self.hidden_size)

    return MockE1Model(mock_hidden_size)


# =============================================================================
# Prototype fixtures
# =============================================================================


@pytest.fixture
def sample_prototypes(mock_hidden_size, device):
    """Sample prototypes [2, hidden_size] for testing."""
    # Create orthogonal prototypes for clear separation
    pos_proto = torch.randn(mock_hidden_size, device=device)
    pos_proto = pos_proto / pos_proto.norm()

    neg_proto = -pos_proto  # Negative is negation of positive

    return torch.stack([neg_proto, pos_proto], dim=0)


@pytest.fixture
def sample_embeddings(mock_hidden_size, device):
    """Sample embeddings for prototype scoring tests."""
    batch_size = 8
    n_views = 4
    return torch.randn(batch_size, n_views, mock_hidden_size, device=device)


@pytest.fixture
def sample_embeddings_2d(mock_hidden_size, device):
    """2D embeddings [batch, hidden] for single-view tests."""
    batch_size = 8
    return torch.randn(batch_size, mock_hidden_size, device=device)


# =============================================================================
# Loss function fixtures
# =============================================================================


@pytest.fixture
def sample_loss_config():
    """Default configuration for loss functions."""
    return {
        "temperature": 0.07,
        "eps": 0.1,
        "eps_pos": 0.25,
        "eps_neg": 0.05,
        "prototype_weight": 1.0,
        "unsupervised_weight": 1.0,
        "bce_weight": 1.0,
        "scoring_temperature": 0.2,
        "label_smoothing": 0.0,
    }


# =============================================================================
# Mock batch preparer fixture
# =============================================================================


@pytest.fixture
def mock_batch_preparer():
    """
    Mock E1BatchPreparer to avoid tokenizer loading issues.

    Returns a callable that produces fake batch kwargs.
    """

    class MockBatchPreparer:
        def __init__(self):
            self.pad_token_id = 0
            self.mask_token_id = 26  # '?' in E1 vocab
            self.vocab = {chr(i + 65): i + 1 for i in range(26)}  # A=1, B=2, ...
            self.vocab.update(
                {"<bos>": 27, "<eos>": 28, "1": 29, "2": 30, "<pad>": 0, "?": 26}
            )
            self.boundary_token_ids = torch.tensor([27, 28, 29, 30, 0])

        def get_batch_kwargs(self, sequences, device=torch.device("cpu")):
            """Generate mock batch dictionaries."""
            batch_size = len(sequences)
            max_len = max(
                len(s.replace(",", "")) + 4 for s in sequences
            )  # +4 for special tokens

            input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
            sequence_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
            within_seq_position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
            global_position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

            for i, seq in enumerate(sequences):
                # Simple tokenization: <bos> 1 A B C ... 2 <eos>
                parts = seq.split(",")
                query = parts[-1] if parts else seq
                tokens = [27, 29] + [self.vocab.get(c, 1) for c in query] + [30, 28]
                seq_len = len(tokens)
                input_ids[i, :seq_len] = torch.tensor(tokens)
                sequence_ids[i, :seq_len] = 0
                within_seq_position_ids[i, :seq_len] = torch.arange(seq_len)
                global_position_ids[i, :seq_len] = torch.arange(seq_len)

            return {
                "input_ids": input_ids.to(device),
                "sequence_ids": sequence_ids.to(device),
                "within_seq_position_ids": within_seq_position_ids.to(device),
                "global_position_ids": global_position_ids.to(device),
            }

    return MockBatchPreparer()


# =============================================================================
# Helper functions
# =============================================================================


def assert_tensor_valid(tensor, name="tensor"):
    """Assert tensor has no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf values"


def assert_shape(tensor, expected_shape, name="tensor"):
    """Assert tensor has expected shape."""
    assert (
        tensor.shape == expected_shape
    ), f"{name} shape {tensor.shape} != expected {expected_shape}"
