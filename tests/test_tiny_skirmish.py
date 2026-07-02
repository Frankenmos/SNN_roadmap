"""Pytest wrappers around the TinySkirmish self-checks (envs/tiny_skirmish).

The env's own self-check modules stay runnable standalone
(`python -m envs.tiny_skirmish.self_check`); these wrappers put them under
pytest/CI. Render and live checks skip when Pillow/pygame are absent
(pygame is not a repo dependency).
"""

import pytest

from envs.tiny_skirmish import self_check


def test_env_self_check():
    assert self_check.main() == 0


def test_torch_adapter_and_harness_self_check():
    from envs.tiny_skirmish import torch_self_check

    assert torch_self_check.main() == 0


def test_render_self_check():
    pytest.importorskip("PIL")
    from envs.tiny_skirmish import render_self_check

    assert render_self_check.main() == 0


def test_live_self_check():
    pytest.importorskip("pygame")
    from envs.tiny_skirmish import live_self_check

    assert live_self_check.main() == 0


def test_real_snn_forward_pass_cpu():
    from envs.tiny_skirmish.real_snn_bridge import run_real_forward_check

    result = run_real_forward_check(seed=9, device_name="cpu", small=True)
    assert result["action_logits_shape"] == (1, 3)


def test_real_snn_rollout_fragment_cpu():
    from envs.tiny_skirmish.real_snn_rollout import collect_real_snn_fragment

    result = collect_real_snn_fragment(
        seed=9,
        max_steps=4,
        device_name="cpu",
        small=True,
    )
    assert result["steps"] >= 1
    assert result["learnable_steps"] >= 1
    assert result["tail_next_batch"] is True
