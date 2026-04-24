"""Tests for CoarseToFineTargetHead."""

import pytest
import torch

from agent_core.target_heads import (
    CoarseToFineTargetHead,
    TargetHeadState,
)
from agent_core.policy_protocol import (
    SPATIAL_OBS_SHAPE,
    META_VECTOR_DIM,
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_RIGHT_CLICK,
)


@pytest.fixture
def head():
    """Create a CoarseToFineTargetHead with standard parameters."""
    return CoarseToFineTargetHead(
        embed_dim=64,
        latent_dim=64,
        action_dim=3,
        coarse_grid_size=7,
        local_grid_size=12,
        screen_size=84,
    )


@pytest.fixture
def dummy_inputs():
    """Create dummy inputs for testing."""
    batch_size = 4
    latent = torch.randn(batch_size, 64)
    spatial_context = torch.randn(batch_size, 64, 7, 7)
    action_ids = torch.tensor([POLICY_ACTION_NO_OP, POLICY_ACTION_RIGHT_CLICK] * 2)
    return latent, spatial_context, action_ids


class TestEncodingDecoding:
    """Test encode/decode roundtrip."""

    def test_encode_xy_to_target_returns_correct_indices(self, head):
        x = torch.tensor([0, 12, 24, 83])
        y = torch.tensor([0, 12, 24, 83])

        result = head.encode_xy_to_target(x, y)

        assert result["coarse_index"] is not None
        assert result["fine_index"] is not None
        assert result["target_index"] is None

        # (0, 0) -> coarse=0 (row 0, col 0), fine=0
        assert result["coarse_index"][0].item() == 0
        assert result["fine_index"][0].item() == 0

        # (12, 12) -> coarse=8 (row 1, col 1), fine=0
        assert result["coarse_index"][1].item() == 8
        assert result["fine_index"][1].item() == 0

        # (24, 24) -> coarse=16 (row 2, col 2), fine=0
        assert result["coarse_index"][2].item() == 16
        assert result["fine_index"][2].item() == 0

        # (83, 83) -> coarse=48 (row 6, col 6), fine=143
        assert result["coarse_index"][3].item() == 48
        assert result["fine_index"][3].item() == 143

    def test_decode_target_to_xy_reconstructs_original(self, head):
        coarse_index = torch.tensor([0, 8, 16, 48])
        fine_index = torch.tensor([0, 1, 2, 143])

        x, y = head.decode_target_to_xy(coarse_index=coarse_index, fine_index=fine_index)

        # (0, 0): coarse=(0,0), fine=(0,0) -> (0, 0)
        assert x[0].item() == 0
        assert y[0].item() == 0

        # (8, 1): coarse=(1,1), fine=(0,1) -> (1*12+1, 1*12+0) = (13, 12)
        assert x[1].item() == 13
        assert y[1].item() == 12

        # (16, 2): coarse=(2,2), fine=(0,2) -> (2*12+2, 2*12+0) = (26, 24)
        assert x[2].item() == 26
        assert y[2].item() == 24

        # (48, 143): coarse=(6,6), fine=(11,11) -> (6*12+11, 6*12+11) = (83, 83)
        assert x[3].item() == 83
        assert y[3].item() == 83

    def test_encode_decode_roundtrip_for_all_positions(self, head):
        """Test exact roundtrip for corner cases."""
        x = torch.tensor([0, 0, 83, 83, 42, 42])
        y = torch.tensor([0, 83, 0, 83, 42, 0])

        encoded = head.encode_xy_to_target(x, y)
        x_recon, y_recon = head.decode_target_to_xy(
            coarse_index=encoded["coarse_index"],
            fine_index=encoded["fine_index"],
        )

        assert torch.all(x == x_recon)
        assert torch.all(y == y_recon)

    def test_encode_clips_to_screen_bounds(self, head):
        x = torch.tensor([-1, 0, 84, 100])
        y = torch.tensor([-1, 0, 84, 100])

        result = head.encode_xy_to_target(x, y)

        # Should clamp to [0, 83]
        assert (result["x"] >= 0).all()
        assert (result["x"] < 84).all()
        assert (result["y"] >= 0).all()
        assert (result["y"] < 84).all()

    def test_decode_requires_both_indices(self, head):
        with pytest.raises(ValueError, match="requires both"):
            head.decode_target_to_xy(coarse_index=torch.tensor([0]))


class TestBuild:
    """Test build() method."""

    def test_build_returns_correct_logit_shapes(self, head, dummy_inputs):
        latent, spatial_context, action_ids = dummy_inputs

        state = head.build(latent, spatial_context, action_ids)

        assert isinstance(state, TargetHeadState)
        assert state.head_type == "coarse_to_fine"
        assert state.primary_logits.shape == (4, 49)  # [B, 7*7]
        assert state.secondary_logits is not None
        assert state.secondary_logits.shape == (4, 49, 144)  # [B, 7*7, 12*12]

    def test_build_with_different_action_ids_changes_logits(self, head, dummy_inputs):
        latent, spatial_context, _ = dummy_inputs

        state_no_op = head.build(
            latent,
            spatial_context,
            torch.full((4,), POLICY_ACTION_NO_OP),
        )
        state_right_click = head.build(
            latent,
            spatial_context,
            torch.full((4,), POLICY_ACTION_RIGHT_CLICK),
        )

        # Logits should differ based on action ID
        assert not torch.allclose(
            state_no_op.primary_logits,
            state_right_click.primary_logits,
        )
        assert not torch.allclose(
            state_no_op.secondary_logits,
            state_right_click.secondary_logits,
        )

    def test_build_validates_grid_size(self):
        head = CoarseToFineTargetHead(
            embed_dim=64,
            latent_dim=64,
            action_dim=3,
            coarse_grid_size=7,
            local_grid_size=12,
            screen_size=84,
        )
        latent = torch.randn(2, 64)
        # Wrong spatial context size
        spatial_context = torch.randn(2, 64, 8, 8)
        action_ids = torch.zeros(2, dtype=torch.long)

        with pytest.raises(ValueError, match="grid mismatch"):
            head.build(latent, spatial_context, action_ids)


class TestSample:
    """Test sample() method."""

    def test_sample_returns_valid_components(self, head, dummy_inputs):
        latent, spatial_context, action_ids = dummy_inputs

        state = head.build(latent, spatial_context, action_ids)
        sample = head.sample(state, deterministic=False)

        assert sample.x.shape == (4,)
        assert sample.y.shape == (4,)
        assert sample.coarse_index is not None
        assert sample.fine_index is not None
        assert sample.coarse_index.shape == (4,)
        assert sample.fine_index.shape == (4,)
        assert sample.log_prob.shape == (4,)
        assert sample.entropy.shape == (4,)

        # Check bounds
        assert (sample.x >= 0).all() and (sample.x < 84).all()
        assert (sample.y >= 0).all() and (sample.y < 84).all()
        assert (sample.coarse_index >= 0).all() and (sample.coarse_index < 49).all()
        assert (sample.fine_index >= 0).all() and (sample.fine_index < 144).all()

    def test_sample_deterministic_uses_argmax(self, head, dummy_inputs):
        latent, spatial_context, action_ids = dummy_inputs

        state = head.build(latent, spatial_context, action_ids)
        sample_det = head.sample(state, deterministic=True)
        sample_stoch = head.sample(state, deterministic=False)

        # Deterministic should match argmax indices
        coarse_argmax = state.primary_logits.argmax(dim=-1)
        assert torch.all(sample_det.coarse_index == coarse_argmax)

        # Fine argmax depends on coarse cell
        batch_idx = torch.arange(4)
        fine_argmax = state.secondary_logits[
            batch_idx,
            sample_det.coarse_index,
        ].argmax(dim=-1)
        assert torch.all(sample_det.fine_index == fine_argmax)

    def test_sample_log_prob_is_sum_of_coarse_and_fine(self, head, dummy_inputs):
        """Verify log_prob = coarse_log_prob + fine_log_prob."""
        latent, spatial_context, action_ids = dummy_inputs

        state = head.build(latent, spatial_context, action_ids)
        sample = head.sample(state, deterministic=True)

        # Manually compute coarse and fine log probs
        coarse_dist = torch.distributions.Categorical(logits=state.primary_logits.float())
        batch_idx = torch.arange(4)
        fine_logits = state.secondary_logits[batch_idx, sample.coarse_index]
        fine_dist = torch.distributions.Categorical(logits=fine_logits.float())

        expected_log_prob = (
            coarse_dist.log_prob(sample.coarse_index)
            + fine_dist.log_prob(sample.fine_index)
        )

        assert torch.allclose(sample.log_prob, expected_log_prob, atol=1e-5)


class TestEvaluate:
    """Test evaluate() method (teacher-forcing)."""

    def testevaluate_uses_recorded_coarse_for_fine(self, head, dummy_inputs):
        """Critical: fine logits must use RECORDED coarse, not resampled."""
        latent, spatial_context, action_ids = dummy_inputs

        state = head.build(latent, spatial_context, action_ids)

        # Recorded targets
        coarse_index = torch.tensor([5, 10, 20, 40])
        fine_index = torch.tensor([0, 50, 100, 143])

        eval_result = head.evaluate(
            state,
            x=torch.zeros(4),
            y=torch.zeros(4),
            coarse_index=coarse_index,
            fine_index=fine_index,
        )

        assert eval_result.log_prob.shape == (4,)
        assert eval_result.entropy.shape == (4,)

        # Verify teacher-forcing: fine logits come from recorded coarse
        coarse_dist = torch.distributions.Categorical(logits=state.primary_logits.float())
        batch_idx = torch.arange(4)
        fine_logits = state.secondary_logits[batch_idx, coarse_index]
        fine_dist = torch.distributions.Categorical(logits=fine_logits.float())

        expected_log_prob = (
            coarse_dist.log_prob(coarse_index)
            + fine_dist.log_prob(fine_index)
        )

        assert torch.allclose(eval_result.log_prob, expected_log_prob, atol=1e-5)

    def testevaluate_infers_indices_from_xy_when_needed(self, head, dummy_inputs):
        latent, spatial_context, action_ids = dummy_inputs

        state = head.build(latent, spatial_context, action_ids)

        x = torch.tensor([0, 12, 24, 83])
        y = torch.tensor([0, 12, 24, 83])

        # Should encode internally
        eval_result = head.evaluate(state, x=x, y=y)

        assert eval_result.log_prob.shape == (4,)


class TestIntegration:
    """Integration tests."""

    def test_full_acting_evaluate_flow(self, head, dummy_inputs):
        """Test acting -> evaluate pipeline like PPO would use."""
        latent, spatial_context, action_ids = dummy_inputs

        # Acting phase
        state = head.build(latent, spatial_context, action_ids)
        sample = head.sample(state, deterministic=False)

        # Evaluation phase (teacher-forcing with recorded targets)
        eval_result = head.evaluate(
            state,
            x=sample.x,
            y=sample.y,
            coarse_index=sample.coarse_index,
            fine_index=sample.fine_index,
        )

        # Log probs should match
        assert torch.allclose(sample.log_prob, eval_result.log_prob, atol=1e-5)

    def test_entropy_magnitude_is_reasonable(self, head, dummy_inputs):
        """Ensure entropy doesn't explode."""
        latent, spatial_context, action_ids = dummy_inputs

        state = head.build(latent, spatial_context, action_ids)
        sample = head.sample(state, deterministic=False)

        # Normalized entropy should be in [0, 1]
        assert (sample.entropy >= 0).all()
        assert (sample.entropy <= 2).all()  # Allow some slack for 2 distributions


def test_coarse_to_fine_properties():
    """Test class properties."""
    head = CoarseToFineTargetHead(
        embed_dim=64,
        latent_dim=64,
        action_dim=3,
        coarse_grid_size=7,
        local_grid_size=12,
        screen_size=84,
    )

    assert head.token_count == 49
    assert head.fine_count == 144
    assert head.head_type == "coarse_to_fine"


def test_coarse_to_fine_validates_config():
    """Test config validation."""
    with pytest.raises(ValueError, match="coarse_grid_size must be positive"):
        CoarseToFineTargetHead(
            embed_dim=64,
            latent_dim=64,
            action_dim=3,
            coarse_grid_size=0,
            local_grid_size=12,
            screen_size=84,
        )

    with pytest.raises(ValueError, match="local_grid_size must be positive"):
        CoarseToFineTargetHead(
            embed_dim=64,
            latent_dim=64,
            action_dim=3,
            coarse_grid_size=7,
            local_grid_size=0,
            screen_size=84,
        )

    with pytest.raises(ValueError, match="coarse_grid_size \\* local_grid_size == screen_size"):
        CoarseToFineTargetHead(
            embed_dim=64,
            latent_dim=64,
            action_dim=3,
            coarse_grid_size=7,
            local_grid_size=10,
            screen_size=84,
        )

    with pytest.raises(ValueError, match="Unsupported target_decode_mode"):
        CoarseToFineTargetHead(
            embed_dim=64,
            latent_dim=64,
            action_dim=3,
            coarse_grid_size=7,
            local_grid_size=12,
            screen_size=84,
            target_decode_mode="invalid",
        )


def test_policy_network_instantiates_with_coarse_to_fine():
    """Smoke test: PolicyNetwork can be created with coarse_to_fine head."""
    from agent_core.spiking_policy import PolicyNetwork
    from agent_core.policy_protocol import SPATIAL_OBS_SHAPE, META_VECTOR_DIM, POLICY_ACTION_DIM

    net = PolicyNetwork(
        SPATIAL_OBS_SHAPE,
        vector_input_dim=META_VECTOR_DIM,
        action_dim=POLICY_ACTION_DIM,
        spatial_head_type="coarse_to_fine",
        num_steps=2,
        screen_size=84,
        attention_embed_dim=64,
        attention_pool_size=7,
    )

    assert net._spatial_head_type == "coarse_to_fine"
    assert isinstance(net.target_head, CoarseToFineTargetHead)
    assert net.target_head.token_count == 49
    assert net.target_head.fine_count == 144


def test_policy_forward_with_coarse_to_fine():
    """Integration test: full forward pass with coarse_to_fine head."""
    from agent_core.spiking_policy import PolicyNetwork
    from agent_core.policy_protocol import (
        SPATIAL_OBS_SHAPE,
        META_VECTOR_DIM,
        POLICY_ACTION_DIM,
    )
    from MockedEnv.policy_batch import make_policy_batch

    net = PolicyNetwork(
        SPATIAL_OBS_SHAPE,
        vector_input_dim=META_VECTOR_DIM,
        action_dim=POLICY_ACTION_DIM,
        spatial_head_type="coarse_to_fine",
        num_steps=2,
        screen_size=84,
        attention_embed_dim=64,
        attention_pool_size=7,
    )
    net.device = torch.device("cpu")
    net.to("cpu")

    # Use the proper batch creation helper
    batch = make_policy_batch(
        batch_size=2,
        spatial_shape=SPATIAL_OBS_SHAPE,
        meta_dim=META_VECTOR_DIM,
        with_state=True,
    )

    action_logits, target_head_state, state_value, next_state = net(batch)

    assert action_logits.shape == (2, POLICY_ACTION_DIM)
    assert target_head_state.primary_logits.shape == (2, 49)  # Coarse
    assert target_head_state.secondary_logits is not None
    assert target_head_state.secondary_logits.shape == (2, 49, 144)  # Fine
    assert state_value.shape == (2,)
    assert len(next_state) == 2

    # Verify we can sample from the head
    sample = net.sample_target(target_head_state, None, deterministic=False)
    assert sample.coarse_index is not None
    assert sample.fine_index is not None
    assert sample.coarse_index.shape == (2,)
    assert sample.fine_index.shape == (2,)
    assert (sample.x >= 0).all() and (sample.x < 84).all()
    assert (sample.y >= 0).all() and (sample.y < 84).all()
