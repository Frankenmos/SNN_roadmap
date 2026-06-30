"""Throwaway diagnostic: reproduce the V6 action logits from a saved eval trace.

Feeds each step's stored (already-normalized) policy_input through the loaded
policy exactly as eval's select_action does, and prints the NO_OP / LEFT /
RIGHT logits + softmax probs so we can see why deterministic argmax = NO_OP
while stochastic sampling attacks.
"""
import glob
import statistics as st

import torch

from Utility.config import cfg
from agent import DefeatRoaches
from agent_core.policy_protocol import (
    PolicyInputBatch,
    POLICY_ACTION_NO_OP,
    POLICY_ACTION_LEFT_CLICK,
    POLICY_ACTION_RIGHT_CLICK,
)

CKPT = "models/banana_glasses_v6_b2048_e4_a10/best_checkpoint.pth"
TRACE_GLOB = "analysis_results/banana_glasses_v6_b2048_e4_a10/episode_traces/episode_000*_det.pt"

NAMES = {
    POLICY_ACTION_NO_OP: "NO_OP",
    POLICY_ACTION_LEFT_CLICK: "LEFT",
    POLICY_ACTION_RIGHT_CLICK: "RIGHT",
}

state = torch.load(CKPT, map_location="cpu", weights_only=False)
agent = DefeatRoaches(
    spatial_input_shape=tuple(cfg.model.spatial_input_shape),
    vector_input_dim=int(cfg.model.vector_input_dim),
    action_dim=int(cfg.model.action_dim),
)
agent.policy.load_state_dict(state["agent_state"])
agent.policy.eval()

# Eval ran at update_count ~1457, where the right-click curriculum penalty is 0.
# A freshly-built agent has update_count=0, which would wrongly subtract 2.0
# from the NO_OP logit. Neutralize it so we measure the trained policy, not the
# curriculum.
agent.ppo.update_count = 10 ** 9

dev = agent.policy.device
amp_dt = agent.policy.amp_dtype
use_amp = agent.policy.use_amp
is_cuda = torch.device(dev).type == "cuda"
print(f"ckpt episode={state.get('episode')} | device={dev} | amp={amp_dt} use_amp={use_amp}")
print(f"curriculum neutralized (update_count={agent.ppo.update_count})\n")


def replay(trace_path, max_steps=12, verbose=True):
    payload = torch.load(trace_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    cur_state = agent.policy.init_concrete_state(
        batch_size=1, device=dev, dtype=torch.float32,
    )
    gaps = []
    value_errs = []
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
        with torch.no_grad():
            with torch.amp.autocast(
                "cuda", dtype=amp_dt, enabled=(use_amp and is_cuda),
            ):
                latent, sv, next_state, _sc = agent.ppo._encode_step_tensors(
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
        if verbose:
            print(
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
    return gaps, value_errs


det_traces = sorted(glob.glob(TRACE_GLOB))
print(f"=== DETAILED: {det_traces[0]} (first 12 policy steps) ===")
replay(det_traces[0], max_steps=12, verbose=True)

print(f"\n=== AGGREGATE: NO_OP-RIGHT masked-logit gap, first 10 steps x {len(det_traces)} det episodes ===")
all_gaps = []
all_verr = []
for tp in det_traces:
    g, verr = replay(tp, max_steps=10, verbose=False)
    all_gaps.extend(g)
    all_verr.extend(verr)
    print(f"  {tp.split(chr(92))[-1]}: mean gap={st.mean(g):+.3f} (min {min(g):+.3f} max {max(g):+.3f})")

print(f"\nOVERALL mean NO_OP-RIGHT gap = {st.mean(all_gaps):+.3f} logits")
print(f"   (positive => NO_OP logit is HIGHER than RIGHT_CLICK => argmax = NO_OP)")
print(f"fidelity check: |my_value - trace_value| mean={st.mean(all_verr):.4f} max={max(all_verr):.4f}")
print(f"   (near 0 => replay reproduces eval's recurrent state faithfully)")
