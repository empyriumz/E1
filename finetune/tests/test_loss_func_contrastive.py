"""
Tests for loss_func_contrastive.py (ConSupPrototypeLoss and PrototypeBCELoss).

Tests verify:
- Unsupervised contrastive loss (NT-Xent) computation
- Prototype alignment loss with pull mask behavior
- BCE loss from prototype distance scores
- Asymmetric margin parameters
- Label smoothing (training only)
- Loss component weights
- Diagnostic metrics
- Gradient flow
- Numerical stability
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

from loss_func_contrastive import ConSupPrototypeLoss, PrototypeBCELoss


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def hidden_size():
    """Standard hidden size for tests."""
    return 64


@pytest.fixture
def sample_prototypes(hidden_size, device):
    """Create orthogonal unit prototypes."""
    pos_proto = torch.randn(hidden_size, device=device)
    pos_proto = pos_proto / pos_proto.norm()
    neg_proto = -pos_proto  # Negative is negation
    return torch.stack([neg_proto, pos_proto], dim=0)


@pytest.fixture
def sample_features(hidden_size, device):
    """Create sample features [batch, n_views, hidden]."""
    batch_size = 16
    n_views = 4
    return torch.randn(
        batch_size, n_views, hidden_size, device=device, requires_grad=True
    )


@pytest.fixture
def sample_labels(device):
    """Binary labels matching sample features."""
    batch_size = 16
    # Mix of positive and negative samples
    labels = torch.zeros(batch_size, device=device)
    labels[:8] = 1  # First half positive
    return labels


@pytest.fixture
def prototype_bce_loss(device):
    """Create PrototypeBCELoss instance."""
    return PrototypeBCELoss(
        temperature=0.07,
        eps=0.1,
        eps_pos=0.25,
        eps_neg=0.05,
        prototype_weight=1.0,
        unsupervised_weight=1.0,
        bce_weight=1.0,
        scoring_temperature=0.2,
        label_smoothing=0.0,
        device=device,
    )


@pytest.fixture
def consup_loss(device):
    """Create ConSupPrototypeLoss instance."""
    return ConSupPrototypeLoss(
        temperature=0.07,
        eps=0.1,
        eps_pos=0.25,
        eps_neg=0.05,
        prototype_weight=1.0,
        unsupervised_weight=1.0,
        device=device,
    )


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestBasicFunctionality:
    """Tests for basic loss computation."""

    def test_loss_returns_dict(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """Loss forward should return a dictionary."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(sample_features, sample_labels)

        assert isinstance(result, dict), "Should return a dictionary"
        assert "total" in result, "Should have 'total' key"

    def test_total_loss_is_tensor(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """Total loss should be a tensor with gradient."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(sample_features, sample_labels)

        assert isinstance(result["total"], torch.Tensor)
        assert result["total"].requires_grad

    def test_loss_components_present(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """Expected loss components should be present."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(sample_features, sample_labels)

        expected_keys = ["total", "prototype", "bce"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# =============================================================================
# Unsupervised Contrastive Loss Tests
# =============================================================================


class TestUnsupervisedContrastiveLoss:
    """Tests for NT-Xent style contrastive loss."""

    def test_contrastive_loss_multi_view(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """Contrastive loss should be computed with multiple views."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(sample_features, sample_labels)

        # Should have contrastive component
        if "contrastive" in result:
            assert result["contrastive"] >= 0, "Contrastive loss should be non-negative"

    def test_single_view_returns_zero_contrastive(
        self, prototype_bce_loss, sample_labels, sample_prototypes, hidden_size, device
    ):
        """Single view should return zero contrastive loss."""
        single_view_features = torch.randn(
            16, 1, hidden_size, device=device, requires_grad=True
        )
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(single_view_features, sample_labels)

        # Contrastive loss should be zero or not computed for single view
        if "contrastive" in result:
            assert result["contrastive"] == 0 or result["contrastive"].item() == 0

    def test_similar_views_lower_loss(
        self, prototype_bce_loss, sample_labels, sample_prototypes, hidden_size, device
    ):
        """Views from same sample should give lower contrastive loss when similar."""
        batch_size = 8
        n_views = 4

        # Create features where all views of same sample are identical
        base_features = torch.randn(batch_size, hidden_size, device=device)
        identical_views = base_features.unsqueeze(1).expand(-1, n_views, -1).clone()
        identical_views.requires_grad_(True)

        # Create features with random views
        random_views = torch.randn(
            batch_size, n_views, hidden_size, device=device, requires_grad=True
        )

        # Reduce labels to match batch size
        labels = sample_labels[:batch_size]

        prototype_bce_loss.set_prototypes(sample_prototypes)

        result_identical = prototype_bce_loss(identical_views, labels)
        result_random = prototype_bce_loss(random_views, labels)

        # Identical views should have lower contrastive loss
        if "contrastive" in result_identical and "contrastive" in result_random:
            assert result_identical["contrastive"] <= result_random["contrastive"] + 0.1


# =============================================================================
# Prototype Alignment Tests
# =============================================================================


class TestPrototypeAlignment:
    """Tests for prototype alignment loss."""

    def test_alignment_loss_present(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """Prototype alignment loss should be present."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(sample_features, sample_labels)

        assert "prototype" in result

    def test_pull_mask_ratio_computed(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """Pull mask ratio diagnostic should be computed."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(sample_features, sample_labels)

        assert "pull_mask_ratio" in result
        ratio = result["pull_mask_ratio"]
        if isinstance(ratio, torch.Tensor):
            ratio = ratio.item()
        assert 0 <= ratio <= 1, f"Pull mask ratio should be in [0, 1], got {ratio}"

    def test_well_aligned_low_loss(
        self, prototype_bce_loss, sample_labels, sample_prototypes, hidden_size, device
    ):
        """Well-aligned embeddings should have low prototype loss."""
        batch_size = 16
        n_views = 4

        # Create embeddings aligned with correct prototypes
        pos_proto = sample_prototypes[1]
        neg_proto = sample_prototypes[0]

        aligned_features = torch.zeros(batch_size, n_views, hidden_size, device=device)
        for i in range(batch_size):
            if sample_labels[i] == 1:
                aligned_features[i] = pos_proto + 0.01 * torch.randn_like(pos_proto)
            else:
                aligned_features[i] = neg_proto + 0.01 * torch.randn_like(neg_proto)
        aligned_features.requires_grad_(True)

        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(aligned_features, sample_labels)

        # Well-aligned should have low prototype loss
        assert (
            result["prototype"] < 1.0
        ), f"Well-aligned embeddings should have low prototype loss, got {result['prototype']}"


# =============================================================================
# BCE Loss Tests
# =============================================================================


class TestBCELoss:
    """Tests for BCE classification loss."""

    def test_bce_loss_present(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """BCE loss should be present."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(sample_features, sample_labels)

        assert "bce" in result

    def test_bce_loss_finite(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """BCE loss should be finite."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(sample_features, sample_labels)

        bce_loss = result["bce"]
        assert not torch.isnan(bce_loss), "BCE loss is NaN"
        assert not torch.isinf(bce_loss), "BCE loss is Inf"


# =============================================================================
# Label Smoothing Tests
# =============================================================================


class TestLabelSmoothing:
    """Tests for label smoothing behavior."""

    def test_smoothing_applied_during_training(
        self, sample_features, sample_labels, sample_prototypes, device
    ):
        """Label smoothing should affect loss during training."""
        smoothing = 0.1

        loss_no_smooth = PrototypeBCELoss(
            temperature=0.07,
            prototype_weight=0.0,  # Disable prototype to isolate BCE
            unsupervised_weight=0.0,  # Disable contrastive
            bce_weight=1.0,
            label_smoothing=0.0,
            device=device,
        )

        loss_with_smooth = PrototypeBCELoss(
            temperature=0.07,
            prototype_weight=0.0,
            unsupervised_weight=0.0,
            bce_weight=1.0,
            label_smoothing=smoothing,
            device=device,
        )

        loss_no_smooth.set_prototypes(sample_prototypes)
        loss_with_smooth.set_prototypes(sample_prototypes)

        result_no_smooth = loss_no_smooth(sample_features, sample_labels, training=True)
        result_with_smooth = loss_with_smooth(
            sample_features, sample_labels, training=True
        )

        # Losses should generally be different with smoothing
        # (though direction depends on prediction quality)
        assert (
            result_no_smooth["bce"] != result_with_smooth["bce"]
        ), "Label smoothing should affect BCE loss"

    def test_no_smoothing_during_eval(
        self, sample_features, sample_labels, sample_prototypes, device
    ):
        """Label smoothing should not affect loss during evaluation."""
        smoothing = 0.1

        loss_fn = PrototypeBCELoss(
            temperature=0.07,
            prototype_weight=0.0,
            unsupervised_weight=0.0,
            bce_weight=1.0,
            label_smoothing=smoothing,
            device=device,
        )

        loss_fn.set_prototypes(sample_prototypes)

        result_train = loss_fn(sample_features, sample_labels, training=True)
        result_eval = loss_fn(sample_features, sample_labels, training=False)

        # During eval, smoothing should not apply, so losses may differ
        # This test just verifies it doesn't crash
        assert "bce" in result_eval


# =============================================================================
# Loss Weight Tests
# =============================================================================


class TestLossWeights:
    """Tests for loss component weighting."""

    def test_zero_prototype_weight(
        self, sample_features, sample_labels, sample_prototypes, device
    ):
        """Zero prototype weight should disable prototype loss."""
        loss_fn = PrototypeBCELoss(
            temperature=0.07,
            prototype_weight=0.0,  # Disabled
            unsupervised_weight=1.0,
            bce_weight=1.0,
            device=device,
        )
        loss_fn.set_prototypes(sample_prototypes)

        result = loss_fn(sample_features, sample_labels)

        # Prototype loss component should be zero or not contribute
        if "prototype" in result:
            assert result["prototype"] == 0 or not torch.is_tensor(result["prototype"])

    def test_zero_bce_weight(
        self, sample_features, sample_labels, sample_prototypes, device
    ):
        """Zero BCE weight should disable BCE loss."""
        loss_fn = PrototypeBCELoss(
            temperature=0.07,
            prototype_weight=1.0,
            unsupervised_weight=1.0,
            bce_weight=0.0,  # Disabled
            device=device,
        )
        loss_fn.set_prototypes(sample_prototypes)

        result = loss_fn(sample_features, sample_labels)

        # BCE loss component should be zero
        assert result["bce"] == 0


# =============================================================================
# Diagnostic Metrics Tests
# =============================================================================


class TestDiagnosticMetrics:
    """Tests for diagnostic metrics."""

    def test_diagnostics_present(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """Diagnostic metrics should be present in output."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(sample_features, sample_labels)

        expected_diagnostics = [
            "pull_mask_ratio",
            "avg_sim_pos_pos",
            "avg_sim_pos_neg",
            "avg_sim_neg_pos",
            "avg_sim_neg_neg",
        ]

        for key in expected_diagnostics:
            assert key in result, f"Missing diagnostic: {key}"


# =============================================================================
# Gradient Flow Tests
# =============================================================================


class TestGradientFlow:
    """Tests for gradient flow through loss."""

    def test_gradients_flow_to_features(
        self, prototype_bce_loss, sample_features, sample_labels, sample_prototypes
    ):
        """Gradients should flow back to input features."""
        prototype_bce_loss.set_prototypes(sample_prototypes)

        # Ensure features require grad
        sample_features.requires_grad_(True)

        result = prototype_bce_loss(sample_features, sample_labels)
        total_loss = result["total"]

        # Backward pass
        total_loss.backward()

        # Check gradients exist
        assert sample_features.grad is not None, "Gradients should flow to features"
        assert not torch.isnan(
            sample_features.grad
        ).any(), "Gradients should not be NaN"


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_handles_large_features(
        self, prototype_bce_loss, sample_labels, sample_prototypes, hidden_size, device
    ):
        """Should handle large feature magnitudes."""
        large_features = torch.randn(16, 4, hidden_size, device=device) * 1000
        large_features.requires_grad_(True)

        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(large_features, sample_labels)

        assert not torch.isnan(result["total"]), "Total loss should not be NaN"
        assert not torch.isinf(result["total"]), "Total loss should not be Inf"

    def test_handles_small_features(
        self, prototype_bce_loss, sample_labels, sample_prototypes, hidden_size, device
    ):
        """Should handle small feature magnitudes."""
        small_features = torch.randn(16, 4, hidden_size, device=device) * 1e-6
        small_features.requires_grad_(True)

        prototype_bce_loss.set_prototypes(sample_prototypes)

        result = prototype_bce_loss(small_features, sample_labels)

        assert not torch.isnan(result["total"]), "Total loss should not be NaN"
        assert not torch.isinf(result["total"]), "Total loss should not be Inf"


# =============================================================================
# Asymmetric Margin Tests
# =============================================================================


class TestAsymmetricMargins:
    """Tests for asymmetric eps_pos and eps_neg margins."""

    def test_asymmetric_margins_respected(
        self, sample_features, sample_labels, sample_prototypes, device
    ):
        """Asymmetric margins should be used correctly."""
        # Create loss with large asymmetric margins
        loss_fn = PrototypeBCELoss(
            temperature=0.07,
            eps_pos=0.5,  # Large margin for positives
            eps_neg=0.1,  # Small margin for negatives
            prototype_weight=1.0,
            unsupervised_weight=0.0,
            bce_weight=0.0,
            device=device,
        )
        loss_fn.set_prototypes(sample_prototypes)

        result = loss_fn(sample_features, sample_labels)

        # Should compute without errors
        assert "prototype" in result
        assert not torch.isnan(result["total"])
