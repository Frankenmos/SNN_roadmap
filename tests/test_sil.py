"""Tests for Self-Imitation Learning (SIL) admission + auxiliary pass."""

import torch
from MockedEnv.policy_batch import make_policy_batch

from agent_core.policy_protocol import (
    ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET,
    META_VECTOR_DIM,
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_RIGHT_CLICK,
    SPATIAL_OBS_SHAPE,
)
from agent_core.ppo_trainer import PPO
from agent_core.spiking_policy import PolicyNetwork


def _make_net(action_dim: int = 3):
    torch.manual_seed(0)
    net = PolicyNetwork(
        SPATIAL_OBS_SHAPE,
        vector_input_dim=META_VECTOR_DIM,
        action_dim=action_dim,
        num_steps=2,
        screen_size=16,
        attention_embed_dim=32,
        attention_pool_size=4,
    )
    net.device = torch.device("cpu")
    net.to("cpu")
    net.use_amp = False
    net.amp_dtype = torch.float32
    net.scaler = torch.amp.GradScaler("cuda", enabled=False)
    return net


def _store_step(
    ppo,
    net,
    *,
    action,
    engagement_feedback,
    reward,
    done,
    sample_mask=1.0,
):
    """Store one transition. ``engagement_feedback`` sets THIS step's feedback
    near-enemy bit. Because feedback[i] describes action[i-1], the engagement
    bit on step i confirms that the PREVIOUS step's action engaged an enemy."""
    state = net.init_concrete_state(batch_size=1, device=torch.device("cpu"))
    batch = make_policy_batch(
        batch_size=1,
        meta_dim=META_VECTOR_DIM,
        with_state=True,
        state_shape=state[0].shape,
    )
    if engagement_feedback:
        batch.action_feedback_tokens[:, :, ACTION_FEEDBACK_TARGET_NEAR_ENEMY_OFFSET] = 1.0
    with torch.no_grad():
        _, _, state_value, _ = net(batch.with_state(state))
    ppo.store_transition(
        batch.with_state(state),
        torch.tensor(int(action)),
        torch.tensor(5),
        torch.tensor(10),
        torch.tensor(-1.0),
        torch.tensor(float(reward)),
        torch.tensor(float(state_value.item())),
        torch.tensor(float(done)),
        sample_mask=torch.tensor(float(sample_mask)),
    )


def _two_attacks_then_idle(ppo, net):
    # step0: RIGHT_CLICK (admitted via step1's engagement feedback)
    _store_step(
        ppo,
        net,
        action=POLICY_ACTION_RIGHT_CLICK,
        engagement_feedback=False,
        reward=50.0,
        done=False,
    )
    # step1: RIGHT_CLICK (its feedback confirms step0 engaged; admitted via step2)
    _store_step(
        ppo,
        net,
        action=POLICY_ACTION_RIGHT_CLICK,
        engagement_feedback=True,
        reward=50.0,
        done=False,
    )
    # step2: NO_OP (its feedback confirms step1 engaged); never admitted itself
    _store_step(
        ppo,
        net,
        action=POLICY_ACTION_NO_OP,
        engagement_feedback=True,
        reward=0.0,
        done=True,
    )


def _log_prob_of_stored_action(ppo, net):
    """Current policy's action-log-prob of the stored action in buffer[0]."""
    entry = ppo.sil_buffer[0]
    obs = entry["observations"]
    with torch.no_grad():
        latent, _, _, _ = ppo._encode_step_tensors(
            spatial_obs=obs.spatial_obs.to(net.device),
            entity_features=obs.entity_features.to(net.device),
            entity_mask=obs.entity_mask.to(net.device),
            selection_features=obs.selection_features.to(net.device),
            selection_mask=obs.selection_mask.to(net.device),
            action_feedback_tokens=obs.action_feedback_tokens.to(net.device),
            meta_vec=obs.meta_vec.to(net.device),
            state_in=entry["initial_state"],
        )
        logits = ppo._mask_action_logits(
            ppo.policy_net.action_head(latent),
            obs.meta_vec.to(net.device),
        )
        dist = torch.distributions.Categorical(logits=logits.float())
        return float(dist.log_prob(entry["actions"].to(net.device)).item())


def test_sil_admits_committed_attacks_and_runs_pass():
    net = _make_net(action_dim=3)
    ppo = PPO(
        net,
        lr=1e-4,
        total_updates=0,
        lr_min=0.0,
        tbptt_window=4,
        sil_enabled=True,
        sil_buffer_size=100,
        sil_batch_fraction=1.0,
        sil_coef=0.5,
    )
    _two_attacks_then_idle(ppo, net)

    assert len(ppo.sil_buffer) == 0  # nothing admitted until an update runs
    losses, stats = ppo.update_policy(batch_size=4, epochs=1)

    # Admission: only the two RIGHT_CLICK steps whose NEXT step confirmed
    # engagement entered the buffer (the NO_OP step is never admitted).
    assert len(ppo.sil_buffer) == 2
    for entry in ppo.sil_buffer:
        assert int(entry["actions"].item()) == POLICY_ACTION_RIGHT_CLICK

    # The SIL auxiliary pass ran and reported diagnostics.
    assert "sil_loss" in stats
    assert "sil_gate_open_fraction" in stats
    assert "sil_buffer_size" in stats
    assert "sil_steps_replayed" in stats
    assert stats["sil_buffer_size"] == 2
    # sil_loss = -coef * mean((R-V)+ . log_prob); (R-V)+ >= 0 and log_prob <= 0
    # so sil_loss is mathematically non-negative.
    assert stats["sil_loss"] >= 0.0
    assert 0.0 <= stats["sil_gate_open_fraction"] <= 1.0
    # High stored returns vs a small random critic => the gate should open.
    assert stats["sil_gate_open_fraction"] > 0.0
    assert stats["sil_steps_replayed"] >= 1
    assert stats["sil_admitted"] == 2
    assert stats["sil_admitted_near_enemy"] == 2
    assert 0 <= stats["sil_admitted_health_drop"] <= 2
    assert stats["sil_admitted_both"] <= stats["sil_admitted_health_drop"]
    assert stats["sil_age_max"] == 0.0
    assert stats["sil_gate_weight_max"] >= stats["sil_gate_weight_mean"]
    assert "sil_grad_norm_trunk" in stats


def test_sil_disabled_leaves_update_unchanged():
    net = _make_net(action_dim=3)
    ppo = PPO(
        net,
        lr=1e-4,
        total_updates=0,
        lr_min=0.0,
        tbptt_window=4,
        sil_enabled=False,
        sil_coef=0.5,
    )
    _two_attacks_then_idle(ppo, net)

    losses, stats = ppo.update_policy(batch_size=4, epochs=1)

    # With SIL off, no admission, no SIL stats, no extra timing keys.
    assert len(ppo.sil_buffer) == 0
    assert not any(k.startswith("sil_") for k in stats)


def test_sil_does_not_admit_masked_right_clicks():
    net = _make_net(action_dim=3)
    ppo = PPO(
        net,
        lr=1e-4,
        total_updates=0,
        lr_min=0.0,
        tbptt_window=4,
        sil_enabled=True,
        sil_buffer_size=100,
        sil_batch_fraction=1.0,
        sil_coef=0.5,
    )
    _store_step(
        ppo,
        net,
        action=POLICY_ACTION_RIGHT_CLICK,
        engagement_feedback=False,
        reward=50.0,
        done=False,
        sample_mask=0.0,
    )
    _store_step(
        ppo,
        net,
        action=POLICY_ACTION_NO_OP,
        engagement_feedback=True,
        reward=0.0,
        done=True,
    )

    losses, stats = ppo.update_policy(batch_size=4, epochs=1)

    assert len(ppo.sil_buffer) == 0
    assert stats["sil_buffer_size"] == 0
    assert stats["sil_steps_replayed"] == 0
    assert stats["sil_groups"] == 0


def test_sil_pass_increases_log_prob_of_stored_good_action():
    """Repeated SIL gradient steps must push the current policy's log-prob of
    the stored good action UP (the entire point of imitation)."""
    net = _make_net(action_dim=3)
    ppo = PPO(
        net,
        lr=5e-3,
        total_updates=0,
        lr_min=0.0,
        tbptt_window=4,
        sil_enabled=True,
        sil_buffer_size=100,
        sil_batch_fraction=1.0,
        sil_coef=2.0,
    )
    # step0: the trophy RIGHT_CLICK (high return); step1: NO_OP whose feedback
    # confirms step0 engaged an enemy, so step0 is admitted.
    _store_step(
        ppo,
        net,
        action=POLICY_ACTION_RIGHT_CLICK,
        engagement_feedback=False,
        reward=50.0,
        done=False,
    )
    _store_step(
        ppo,
        net,
        action=POLICY_ACTION_NO_OP,
        engagement_feedback=True,
        reward=0.0,
        done=True,
    )

    # First update admits the transition and runs one SIL pass.
    ppo.update_policy(batch_size=4, epochs=1)
    assert len(ppo.sil_buffer) == 1
    log_prob_after = _log_prob_of_stored_action(ppo, net)

    params = [p for p in net.parameters() if p.requires_grad]
    for _ in range(10):
        ppo._run_sil_pass(rollout_size=4, params=params, timings=None)

    log_prob_later = _log_prob_of_stored_action(ppo, net)
    assert log_prob_later > log_prob_after
