"""Probe what a trained policy's action head actually believes.

Replays a saved eval trace's per-step (already-normalized) policy_inputs through
the loaded policy exactly as eval's `select_action` does, and prints the
NO_OP / LEFT / RIGHT action logits + softmax probabilities + the current value.
Use this to see why deterministic eval idles (argmax = NO_OP) while stochastic
eval attacks, and to measure whether SIL / reward changes move P(RIGHT).

Replay is faithful: each step is fed with its stored pre-step SNN state, and the
recomputed value is checked against the value the trace recorded (they should
match to ~0, confirming the recurrent state is reproduced).

Example:
    python tools/analysis/probe_action_logits.py --run-name banana_glasses_v6_b2048_e4_a10 --best
"""
from __future__ import annotations

import argparse
import glob
import statistics as st
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from Utility.config import cfg
from agent import DefeatRoaches
from agent_core.policy_protocol import (
    PolicyInputBatch,
    POLICY_ACTION_LEFT_CLICK,
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_RIGHT_CLICK,
)


NAMES = {
    POLICY_ACTION_NO_OP: "NO_OP",
    POLICY_ACTION_LEFT_CLICK: "LEFT",
    POLICY_ACTION_RIGHT_CLICK: "RIGHT",
}


def locate_checkpoint(args) -> str:
    if args.checkpoint:
        return args.checkpoint
    name = args.run_name or getattr(cfg.environment, "run_name", "")
    if not name:
        raise SystemExit("No --checkpoint and no run_name; pass one.")
    models_dir = getattr(cfg.environment, "models_dir", "models")
    filename = (
        getattr(cfg.environment, "best_checkpoint_path", "best_checkpoint.pth")
        if args.best
        else getattr(cfg.environment, "checkpoint_path", "checkpoint.pth")
    )
    return str(Path(models_dir) / name / filename)


def default_traces(args) -> list[str]:
    if args.traces:
        return sorted(glob.glob(args.traces))
    name = args.run_name or getattr(cfg.environment, "run_name", "")
    analysis_dir = getattr(cfg.environment, "analysis_dir", "analysis_results")
    pattern = str(Path(analysis_dir) / name / "episode_traces" / "episode_*_det.pt")
    return sorted(glob.glob(pattern))


def build_agent(checkpoint_path: str):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = DefeatRoaches(
        spatial_input_shape=tuple(cfg.model.spatial_input_shape),
        vector_input_dim=int(cfg.model.vector_input_dim),
        action_dim=int(cfg.model.action_dim),
    )
    agent.policy.load_state_dict(state["agent_state"])
    agent.policy.eval()
    # Eval ran at a late update_count where the right-click curriculum penalty
    # is 0; a freshly-built agent starts at 0 which would wrongly suppress the
    # NO_OP logit. Neutralize it so we measure the trained policy, not curriculum.
    agent.ppo.update_count = 10 ** 9
    return agent, state


def replay_trace(agent, trace_path, max_steps):
    dev = agent.policy.device
    amp_dt = agent.policy.amp_dtype
    use_amp = agent.policy.use_amp
    is_cuda = torch.device(dev).type == "cuda"
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    cur_state = agent.policy.init_concrete_state(
        batch_size=1, device=dev, dtype=torch.float32,
    )
    gaps, value_errs, lines = [], [], []
    n = 0
    for rec in records:
        serial = rec.get("policy_input")
        if serial is None:
            continue  # bootstrap select_army step
        batch = PolicyInputBatch(
            spatial_obs=serial["spatial_obs"].to(dev).float().unsqueeze(0),
            entity_features=serial["entity_features"].to(dev).float().unsqueeze(0),
            entity_mask=serial["entity_mask"].to(dev).unsqueeze(0),
            selection_features=serial["selection_features"].to(dev).float().unsqueeze(0),
            selection_mask=serial["selection_mask"].to(dev).unsqueeze(0),
            action_feedback_tokens=serial["action_feedback_tokens"].to(dev).float().unsqueeze(0),
            meta_vec=serial["meta_vec"].to(dev).float().unsqueeze(0),
            state_in=cur_state,
        )
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=amp_dt, enabled=(use_amp and is_cuda),
        ):
            latent, sv, next_state, _ = agent.ppo._encode_step_tensors(
                spatial_obs=batch.spatial_obs,
                entity_features=batch.entity_features,
                entity_mask=batch.entity_mask,
                selection_features=batch.selection_features,
                selection_mask=batch.selection_mask,
                action_feedback_tokens=batch.action_feedback_tokens,
                meta_vec=batch.meta_vec,
                state_in=batch.state_in,
            )
            raw = agent.policy.action_head(latent).float()
            masked = agent.ppo._mask_action_logits(raw, batch.meta_vec).float()
        probs = torch.softmax(masked, dim=-1)[0]
        m = masked[0].tolist()
        am = int(masked[0].argmax().item())
        no_op, right = m[POLICY_ACTION_NO_OP], m[POLICY_ACTION_RIGHT_CLICK]
        gaps.append(no_op - right)
        computed_v = float(sv.reshape(-1)[0].item())
        trace_v = float(rec.get("value") or 0.0)
        value_errs.append(abs(computed_v - trace_v))
        lines.append(
            f"  step {rec['step_index']:3d} | masked[N,L,R]="
            f"[{m[0]:+.3f},{m[1]:+.3f},{m[2]:+.3f}] "
            f"| P(NO_OP)={probs[POLICY_ACTION_NO_OP]:.3f} "
            f"P(RIGHT)={probs[POLICY_ACTION_RIGHT_CLICK]:.3f} "
            f"| argmax={NAMES[am]:5s} "
            f"| V: mine={computed_v:+.3f} trace={trace_v:+.3f} "
            f"| trace_action={rec.get('action')}"
        )
        cur_state = next_state
        n += 1
        if n >= max_steps:
            break
    return gaps, value_errs, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", help="Explicit path to a .pth file.")
    parser.add_argument(
        "--run-name", help="Run dir under models/ and analysis_results/.",
    )
    parser.add_argument("--best", action="store_true", help="Prefer best_checkpoint.pth.")
    parser.add_argument("--traces", help="Glob for det trace .pt files.")
    parser.add_argument("--episodes", type=int, default=0, help="Trace files to scan (0=all).")
    parser.add_argument("--detail-steps", type=int, default=12, help="Per-step printout for the first trace.")
    parser.add_argument("--agg-steps", type=int, default=10, help="Steps per trace for the aggregate gap.")
    args = parser.parse_args()

    checkpoint_path = locate_checkpoint(args)
    traces = default_traces(args)
    if not traces:
        raise SystemExit("No det traces found. Run eval with --trace_episodes first.")
    if args.episodes:
        traces = traces[: args.episodes]

    agent, state = build_agent(checkpoint_path)
    dev = agent.policy.device
    print(
        f"checkpoint={checkpoint_path} | episode={state.get('episode')} "
        f"| device={dev} | amp={agent.policy.amp_dtype} "
        f"| traces={len(traces)}\n"
    )

    print(f"=== DETAILED: {traces[0]} (first {args.detail_steps} policy steps) ===")
    _, _, first_lines = replay_trace(agent, traces[0], max_steps=args.detail_steps)
    for line in first_lines:
        print(line)

    print(
        f"\n=== AGGREGATE: NO_OP-RIGHT masked-logit gap, "
        f"first {args.agg_steps} steps x {len(traces)} det episodes ===",
    )
    all_gaps, all_verr = [], []
    for tp in traces:
        g, verr, _ = replay_trace(agent, tp, max_steps=args.agg_steps)
        all_gaps.extend(g)
        all_verr.extend(verr)
        print(
            f"  {Path(tp).name}: mean gap={st.mean(g):+.3f} "
            f"(min {min(g):+.3f} max {max(g):+.3f})",
        )

    print(f"\nOVERALL mean NO_OP-RIGHT gap = {st.mean(all_gaps):+.3f} logits")
    print("   (positive => NO_OP logit higher than RIGHT_CLICK => argmax = NO_OP)")
    print(
        f"fidelity: |my_value - trace_value| mean={st.mean(all_verr):.4f} "
        f"max={max(all_verr):.4f} (near 0 => replay reproduces eval state)",
    )


if __name__ == "__main__":
    main()
