import json

import torch

from MockedEnv.policy_batch import make_policy_batch
from agent_core.policy_protocol import META_VECTOR_DIM, SPATIAL_OBS_SHAPE
from agent_core.spiking_policy import PolicyNetwork
from tools.analysis.analyze_eval_trace import EvalTraceAnalyzer


def test_eval_trace_analyzer_exports_basic_bundle(tmp_path):
    batch = make_policy_batch(
        batch_size=1,
        spatial_shape=SPATIAL_OBS_SHAPE,
        meta_dim=META_VECTOR_DIM,
        zeros=True,
    ).with_state(None)

    trace_path = tmp_path / "episode_0001_det.pt"
    torch.save(
        {
            "format_version": 1,
            "run_name": "dummy-run",
            "checkpoint_path": "models/dummy-run/best_checkpoint.pth",
            "checkpoint_episode": 321,
            "deterministic": True,
            "episode_index": 1,
            "total_reward": 2.5,
            "steps": 3,
            "records": [
                {
                    "step_index": 0,
                    "action": None,
                    "move_x": 0,
                    "move_y": 0,
                    "reward": 0.0,
                    "cumulative_reward": 0.0,
                    "done": False,
                    "learnable": False,
                    "policy_step": False,
                    "dispatched_action": {
                        "function_id": 7,
                        "function_name": "select_army",
                        "arguments": [[0]],
                    },
                    "policy_input": None,
                },
                {
                    "step_index": 1,
                    "action": 1,
                    "move_x": 12,
                    "move_y": 34,
                    "reward": 1.0,
                    "cumulative_reward": 1.0,
                    "done": False,
                    "learnable": True,
                    "policy_step": True,
                    "dispatched_action": {
                        "function_id": 451,
                        "function_name": "Smart_screen",
                        "arguments": ["now", [12, 34]],
                    },
                    "policy_input": {
                        "spatial_obs": batch.spatial_obs[0].clone(),
                        "entity_features": batch.entity_features[0].clone(),
                        "entity_mask": batch.entity_mask[0].clone(),
                        "selection_features": batch.selection_features[0].clone(),
                        "selection_mask": batch.selection_mask[0].clone(),
                        "meta_vec": batch.meta_vec[0].clone(),
                    },
                },
                {
                    "step_index": 2,
                    "action": 0,
                    "move_x": 20,
                    "move_y": 21,
                    "reward": 1.5,
                    "cumulative_reward": 2.5,
                    "done": True,
                    "learnable": True,
                    "policy_step": True,
                    "dispatched_action": {
                        "function_id": 0,
                        "function_name": "no_op",
                        "arguments": [],
                    },
                    "policy_input": {
                        "spatial_obs": batch.spatial_obs[0].clone(),
                        "entity_features": batch.entity_features[0].clone(),
                        "entity_mask": batch.entity_mask[0].clone(),
                        "selection_features": batch.selection_features[0].clone(),
                        "selection_mask": batch.selection_mask[0].clone(),
                        "meta_vec": batch.meta_vec[0].clone(),
                    },
                },
            ],
        },
        trace_path,
    )

    analyzer = EvalTraceAnalyzer(trace_path)
    summary = analyzer.summarize()
    out_dir = tmp_path / "bundle"
    exported = analyzer.export_panels(out_dir)

    assert summary["bootstrap_steps"] == 1
    assert summary["policy_steps"] == 2
    assert summary["action_counts"]["smart"] == 1
    assert summary["action_counts"]["no-op"] == 1
    assert "trace_report.txt" in exported
    assert "05_spatial_planes.png" in exported
    assert (out_dir / "manifest.txt").exists()
    assert (out_dir / "04_spatial_targets.png").exists()
    report_text = (out_dir / "trace_report.txt").read_text(encoding="utf-8")
    assert "Dispatched action counts:" in report_text
    assert "Smart_screen" in report_text


def test_eval_trace_analyzer_labels_coarse_to_fine_semantic_clicks(tmp_path):
    ckpt_path = tmp_path / "best_checkpoint.pth"
    ckpt_path.write_bytes(b"checkpoint placeholder")
    (tmp_path / "effective_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "action_dim": 3,
                    "spatial_head_type": "coarse_to_fine",
                },
            },
        ),
        encoding="utf-8",
    )
    trace_path = tmp_path / "episode_0001_det.pt"
    torch.save(
        {
            "format_version": 1,
            "run_name": "coarse-run",
            "checkpoint_path": str(ckpt_path),
            "records": [
                {
                    "step_index": 1,
                    "action": 2,
                    "policy_step": True,
                    "learnable": True,
                    "reward": 0.0,
                    "cumulative_reward": 0.0,
                    "dispatched_action": {
                        "function_id": 451,
                        "function_name": "Smart_screen",
                    },
                    "policy_input": {},
                },
            ],
        },
        trace_path,
    )

    analyzer = EvalTraceAnalyzer(trace_path)
    summary = analyzer.summarize()

    assert analyzer.action_labels[0] == "no-op"
    assert analyzer.action_labels[1] == "left_click"
    assert analyzer.action_labels[2] == "right_click"
    assert summary["action_counts"]["right_click"] == 1
    assert "attack" not in summary["action_counts"]


def test_eval_trace_activation_policy_rebuild_uses_target_head_config(tmp_path):
    batch = make_policy_batch(
        batch_size=1,
        spatial_shape=SPATIAL_OBS_SHAPE,
        meta_dim=META_VECTOR_DIM,
        zeros=True,
    ).with_state(None)
    model_cfg = {
        "action_dim": 3,
        "num_steps": 1,
        "screen_size": 84,
        "attention_embed_dim": 64,
        "attention_pool_size": 7,
        "spatial_head_type": "coarse_to_fine",
        "coarse_grid_size": 7,
        "local_grid_size": 12,
        "target_decode_mode": "center",
        "fine_skip_connection": True,
        "fine_skip_dim": 8,
    }
    policy = PolicyNetwork(
        spatial_input_shape=SPATIAL_OBS_SHAPE,
        vector_input_dim=META_VECTOR_DIM,
        **model_cfg,
    )
    ckpt_path = tmp_path / "best_checkpoint.pth"
    torch.save({"agent_state": policy.state_dict()}, ckpt_path)
    (tmp_path / "effective_config.json").write_text(
        json.dumps({"model": model_cfg}),
        encoding="utf-8",
    )
    trace_path = tmp_path / "episode_0001_det.pt"
    torch.save(
        {
            "format_version": 1,
            "run_name": "coarse-run",
            "checkpoint_path": str(ckpt_path),
            "records": [
                {
                    "step_index": 1,
                    "action": 2,
                    "policy_step": True,
                    "learnable": True,
                    "reward": 0.0,
                    "cumulative_reward": 0.0,
                    "dispatched_action": {
                        "function_id": 451,
                        "function_name": "Smart_screen",
                    },
                    "policy_input": {
                        "spatial_obs": batch.spatial_obs[0].clone(),
                        "entity_features": batch.entity_features[0].clone(),
                        "entity_mask": batch.entity_mask[0].clone(),
                        "selection_features": batch.selection_features[0].clone(),
                        "selection_mask": batch.selection_mask[0].clone(),
                        "meta_vec": batch.meta_vec[0].clone(),
                    },
                },
            ],
        },
        trace_path,
    )

    analyzer = EvalTraceAnalyzer(trace_path)
    rebuilt = analyzer._instantiate_policy(ckpt_path)

    assert rebuilt._spatial_head_type == "coarse_to_fine"
    assert rebuilt._coarse_grid_size == 7
    assert rebuilt._local_grid_size == 12
    assert rebuilt._target_decode_mode == "center"
    assert rebuilt._fine_skip_connection is True
    assert rebuilt._fine_skip_dim == 8
