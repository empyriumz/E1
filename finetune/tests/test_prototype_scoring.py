"""
Tests for prototype_scoring.py module.

Tests verify:
- Score computation for 2D and 3D embeddings
- Correct formula: score = sim(x, pos_proto) - sim(x, neg_proto)
- Temperature scaling effect
- Prototype normalization
- Probability computation in valid range
- NaN/Inf handling
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Add finetune directory to path
finetune_dir = Path(__file__).parent.parent
if str(finetune_dir) not in sys.path:
    sys.path.insert(0, str(finetune_dir))

from training.prototype_scoring import (
    compute_prototype_distance_scores,
    compute_prototype_probabilities,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def hidden_size():
    """Standard hidden size for testing."""
    return 64


@pytest.fixture
def unit_prototypes(hidden_size, device):
    """Create unit-normalized orthogonal prototypes."""
    # Positive prototype
    pos_proto = torch.randn(hidden_size, device=device)
    pos_proto = pos_proto / pos_proto.norm()

    # Negative prototype = -positive (as per E1 design)
    neg_proto = -pos_proto

    return torch.stack([neg_proto, pos_proto], dim=0)


@pytest.fixture
def aligned_embeddings(unit_prototypes, device):
    """Create embeddings that are aligned with prototypes."""
    pos_proto = unit_prototypes[1]
    neg_proto = unit_prototypes[0]

    # Create embeddings: some close to pos, some close to neg
    batch_size = 10
    embeddings = []
    labels = []

    for i in range(batch_size):
        if i < 5:
            # Aligned with positive prototype
            emb = pos_proto + 0.1 * torch.randn_like(pos_proto)
            labels.append(1)
        else:
            # Aligned with negative prototype
            emb = neg_proto + 0.1 * torch.randn_like(neg_proto)
            labels.append(0)
        embeddings.append(emb)

    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels)
    return embeddings, labels


# =============================================================================
# 2D Embeddings Tests
# =============================================================================


class Test2DEmbeddings:
    """Tests for 2D embeddings [batch, feat_dim]."""

    def test_output_shape(self, hidden_size, unit_prototypes, device):
        """Output should have shape [batch_size]."""
        batch_size = 8
        embeddings = torch.randn(batch_size, hidden_size, device=device)

        scores = compute_prototype_distance_scores(embeddings, unit_prototypes)

        assert scores.shape == (
            batch_size,
        ), f"Expected shape ({batch_size},), got {scores.shape}"

    def test_scores_are_finite(self, hidden_size, unit_prototypes, device):
        """Scores should not contain NaN or Inf."""
        embeddings = torch.randn(8, hidden_size, device=device)

        scores = compute_prototype_distance_scores(embeddings, unit_prototypes)

        assert not torch.isnan(scores).any(), "Scores contain NaN"
        assert not torch.isinf(scores).any(), "Scores contain Inf"


# =============================================================================
# 3D Embeddings Tests (Multi-Variant)
# =============================================================================


class Test3DEmbeddings:
    """Tests for 3D embeddings [batch, n_views, feat_dim]."""

    def test_output_shape(self, hidden_size, unit_prototypes, device):
        """Output should have shape [batch_size] (averaged over views)."""
        batch_size = 8
        n_views = 4
        embeddings = torch.randn(batch_size, n_views, hidden_size, device=device)

        scores = compute_prototype_distance_scores(embeddings, unit_prototypes)

        assert scores.shape == (
            batch_size,
        ), f"Expected shape ({batch_size},), got {scores.shape}"

    def test_multi_view_averaging(self, hidden_size, unit_prototypes, device):
        """Scores should be averaged across views."""
        batch_size = 4
        n_views = 2

        # Create embeddings where view 0 is close to pos, view 1 is close to neg
        pos_proto = unit_prototypes[1]
        neg_proto = unit_prototypes[0]

        embeddings = torch.zeros(batch_size, n_views, hidden_size, device=device)
        embeddings[:, 0, :] = pos_proto  # View 0 close to positive
        embeddings[:, 1, :] = neg_proto  # View 1 close to negative

        scores = compute_prototype_distance_scores(embeddings, unit_prototypes)

        # Average of high (view 0) and low (view 1) scores should be near zero
        assert (
            torch.abs(scores.mean()) < 0.5
        ), f"Averaged scores should be near zero, got {scores.mean()}"


# =============================================================================
# Score Computation Tests
# =============================================================================


class TestScoreComputation:
    """Tests for score computation formula."""

    def test_score_formula(self, hidden_size, device):
        """Verify score = sim(x, pos) - sim(x, neg)."""
        # Create simple prototypes
        pos_proto = torch.zeros(hidden_size, device=device)
        pos_proto[0] = 1.0  # Unit vector in first dimension

        neg_proto = -pos_proto
        prototypes = torch.stack([neg_proto, pos_proto], dim=0)

        # Create embedding aligned with positive
        emb = torch.zeros(1, hidden_size, device=device)
        emb[0, 0] = 1.0  # Same direction as pos_proto

        scores = compute_prototype_distance_scores(
            emb, prototypes, scoring_temperature=1.0
        )

        # Expected: sim_pos = 1.0, sim_neg = -1.0, score = 1.0 - (-1.0) = 2.0
        expected = 2.0
        assert torch.isclose(
            scores[0], torch.tensor(expected), atol=0.01
        ), f"Expected score {expected}, got {scores[0].item()}"

    def test_positive_samples_get_positive_scores(
        self, unit_prototypes, aligned_embeddings, device
    ):
        """Embeddings aligned with pos_proto should get positive scores."""
        embeddings, labels = aligned_embeddings

        scores = compute_prototype_distance_scores(embeddings, unit_prototypes)

        pos_mask = labels == 1
        pos_scores = scores[pos_mask]

        # Positive samples should have positive scores (closer to pos_proto)
        assert (
            pos_scores > 0
        ).all(), f"Positive samples should have positive scores, got {pos_scores}"

    def test_negative_samples_get_negative_scores(
        self, unit_prototypes, aligned_embeddings, device
    ):
        """Embeddings aligned with neg_proto should get negative scores."""
        embeddings, labels = aligned_embeddings

        scores = compute_prototype_distance_scores(embeddings, unit_prototypes)

        neg_mask = labels == 0
        neg_scores = scores[neg_mask]

        # Negative samples should have negative scores (closer to neg_proto)
        assert (
            neg_scores < 0
        ).all(), f"Negative samples should have negative scores, got {neg_scores}"


# =============================================================================
# Temperature Scaling Tests
# =============================================================================


class TestTemperatureScaling:
    """Tests for scoring temperature effect."""

    def test_temperature_scales_scores(self, hidden_size, unit_prototypes, device):
        """Lower temperature should produce larger absolute scores."""
        embeddings = torch.randn(8, hidden_size, device=device)

        scores_temp_1 = compute_prototype_distance_scores(
            embeddings, unit_prototypes, scoring_temperature=1.0
        )
        scores_temp_low = compute_prototype_distance_scores(
            embeddings, unit_prototypes, scoring_temperature=0.1
        )

        # Lower temperature should amplify scores
        assert (
            scores_temp_low.abs().mean() > scores_temp_1.abs().mean()
        ), "Lower temperature should produce larger absolute scores"

    def test_high_temperature_compresses_scores(
        self, hidden_size, unit_prototypes, device
    ):
        """Higher temperature should compress score range."""
        embeddings = torch.randn(8, hidden_size, device=device)

        scores_temp_1 = compute_prototype_distance_scores(
            embeddings, unit_prototypes, scoring_temperature=1.0
        )
        scores_temp_high = compute_prototype_distance_scores(
            embeddings, unit_prototypes, scoring_temperature=5.0
        )

        # Higher temperature should reduce score magnitude
        assert (
            scores_temp_high.abs().mean() < scores_temp_1.abs().mean()
        ), "Higher temperature should compress scores"


# =============================================================================
# Probability Computation Tests
# =============================================================================


class TestProbabilityComputation:
    """Tests for probability computation."""

    def test_probabilities_in_valid_range(self, hidden_size, unit_prototypes, device):
        """Probabilities should be in [0, 1]."""
        embeddings = torch.randn(100, hidden_size, device=device)

        probs = compute_prototype_probabilities(embeddings, unit_prototypes)

        assert (probs >= 0).all(), "Probabilities should be >= 0"
        assert (probs <= 1).all(), "Probabilities should be <= 1"

    def test_positive_aligned_get_high_probability(
        self, unit_prototypes, aligned_embeddings, device
    ):
        """Embeddings close to pos_proto should get high probability."""
        embeddings, labels = aligned_embeddings

        probs = compute_prototype_probabilities(embeddings, unit_prototypes)

        pos_mask = labels == 1
        pos_probs = probs[pos_mask]

        # Positive samples should have probability > 0.5
        assert (
            pos_probs > 0.5
        ).all(), f"Positive samples should have prob > 0.5, got {pos_probs}"

    def test_negative_aligned_get_low_probability(
        self, unit_prototypes, aligned_embeddings, device
    ):
        """Embeddings close to neg_proto should get low probability."""
        embeddings, labels = aligned_embeddings

        probs = compute_prototype_probabilities(embeddings, unit_prototypes)

        neg_mask = labels == 0
        neg_probs = probs[neg_mask]

        # Negative samples should have probability < 0.5
        assert (
            neg_probs < 0.5
        ).all(), f"Negative samples should have prob < 0.5, got {neg_probs}"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_prototype_count_raises(self, hidden_size, device):
        """Should raise error if not exactly 2 prototypes."""
        embeddings = torch.randn(8, hidden_size, device=device)
        prototypes = torch.randn(3, hidden_size, device=device)  # Wrong count

        with pytest.raises(ValueError, match="Expected 2 prototypes"):
            compute_prototype_distance_scores(embeddings, prototypes)

    def test_none_prototypes_raises(self, hidden_size, device):
        """Should raise error if prototypes is None."""
        embeddings = torch.randn(8, hidden_size, device=device)

        with pytest.raises(ValueError, match="Prototypes must be provided"):
            compute_prototype_distance_scores(embeddings, None)

    def test_nan_embeddings_return_zeros(self, hidden_size, unit_prototypes, device):
        """NaN embeddings should return zero scores (graceful handling)."""
        embeddings = torch.full((8, hidden_size), float("nan"), device=device)

        scores = compute_prototype_distance_scores(embeddings, unit_prototypes)

        # Should return zeros, not crash
        assert (scores == 0).all(), "NaN embeddings should produce zero scores"


# =============================================================================
# Normalization Tests
# =============================================================================


class TestNormalization:
    """Tests for embedding normalization."""

    def test_unnormalized_embeddings_handled(
        self, hidden_size, unit_prototypes, device
    ):
        """Unnormalized embeddings should still produce valid scores."""
        # Create embeddings with large magnitudes
        embeddings = torch.randn(8, hidden_size, device=device) * 100

        scores = compute_prototype_distance_scores(embeddings, unit_prototypes)

        # Scores should still be finite (normalization happens internally)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()

        # Score range should be bounded (after normalization, max diff is 2)
        # With temperature=1.0, scores should be in [-2, 2]
        assert (
            scores.abs() <= 2.1
        ).all(), f"Scores should be bounded, got max {scores.abs().max()}"
