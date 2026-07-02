"""Stage 0 analysis: V5 deterministic eval diagnostics.

Checks: action distribution, Smart click coordinates (corner-lock?),
coarse/fine index diversity (static-prior evidence), feedback-derived
target_near_enemy / damage rates, episode rewards.
"""
import json
import os
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    path = os.path.join(BASE, name)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ---------- 1) dispatched actions ----------
func_counts = Counter()
smart_coords = []          # (episode, step, x, y)
for rec in load("last_action_diagnostics_det.jsonl"):
    da = rec.get("dispatched_action")
    if not da:
        continue
    func_counts[(da["function_id"], da["function_name"])] += 1
    if da["function_id"] == 451:  # Smart_screen
        args = da.get("arguments", [])
        # args = [[queued], [x, y]]
        if len(args) >= 2 and len(args[1]) >= 2:
            x, y = args[1][0], args[1][1]
            smart_coords.append((rec.get("episode"), rec.get("step"), x, y))

print("=== Dispatched function distribution ===")
total = sum(func_counts.values())
for (fid, name), c in func_counts.most_common():
    print(f"  {fid:>4} {name:<22} {c:>6}  ({100*c/total:.1f}%)")
print(f"  total dispatched steps: {total}")

print()
print("=== Smart_screen click coordinates ===")
print(f"  smart clicks: {len(smart_coords)}")
coord_counts = Counter((x, y) for _, _, x, y in smart_coords)
print(f"  unique (x,y): {len(coord_counts)}")
for (x, y), c in coord_counts.most_common(15):
    cell_cx, cell_cy = x // 12, y // 12
    fine_fx, fine_fy = x % 12, y % 12
    coarse_idx = cell_cy * 7 + cell_cx
    fine_idx = fine_fy * 12 + fine_fx
    print(f"  ({x:>2},{y:>2}) x{c:<5} coarse_cell=({cell_cx},{cell_cy}) idx={coarse_idx:<2} fine=({fine_fx},{fine_fy}) idx={fine_idx}")

coarse_counts = Counter(((y // 12) * 7 + (x // 12)) for _, _, x, y in smart_coords)
fine_counts = Counter(((y % 12) * 12 + (x % 12)) for _, _, x, y in smart_coords)
print(f"  unique coarse cells used: {len(coarse_counts)} / 49 -> {dict(coarse_counts.most_common(10))}")
print(f"  unique fine sub-indices used: {len(fine_counts)} / 144 -> {dict(fine_counts.most_common(10))}")

# per-episode click summary
eps = sorted(set(e for e, _, _, _ in smart_coords))
print("  per-episode click spread:")
for e in eps:
    pts = [(x, y) for ee, _, x, y in smart_coords if ee == e]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    print(f"    ep{e}: n={len(pts)} x[{min(xs)}-{max(xs)}] y[{min(ys)}-{max(ys)}] unique={len(set(pts))}")

# ---------- 2) feedback-derived outcome rates ----------
n_exec_smart = 0
near_enemy = 0
enemy_dmg = 0
friendly_toward = 0
effect_classes = Counter()
for rec in load("policy_input_diagnostics_det.jsonl"):
    batch = rec.get("batch") or {}
    fb = batch.get("action_feedback_named")
    if not fb:
        continue
    effect_classes[batch.get("action_feedback_effect_class")] += 1
    if fb.get("executed_smart", 0) >= 0.5:
        n_exec_smart += 1
        near_enemy += fb.get("target_near_enemy", 0) >= 0.5
        enemy_dmg += fb.get("enemy_health_drop_norm", 0) > 0
        friendly_toward += fb.get("friendly_moved_toward_target", 0) >= 0.5

print()
print("=== Feedback-derived Smart outcomes (env v2 feedback) ===")
print(f"  steps with executed_smart=1: {n_exec_smart}")
if n_exec_smart:
    print(f"  target_near_enemy rate:        {near_enemy}/{n_exec_smart} ({100*near_enemy/n_exec_smart:.1f}%)")
    print(f"  enemy_health_drop>0 rate:      {enemy_dmg}/{n_exec_smart} ({100*enemy_dmg/n_exec_smart:.1f}%)")
    print(f"  friendly_moved_toward_target:  {friendly_toward}/{n_exec_smart} ({100*friendly_toward/n_exec_smart:.1f}%)")
print(f"  effect_class distribution: {dict(effect_classes.most_common())}")

# ---------- 3) episode rewards ----------
print()
print("=== Episode rewards (score diagnostics) ===")
ep_last = {}
for rec in load("score_diagnostics_det.jsonl"):
    e = rec.get("episode")
    ep_last[e] = rec  # last record per episode wins
for e in sorted(ep_last):
    r = ep_last[e]
    sc = r.get("score_cumulative_named") or {}
    print(f"  ep{e}: episode_reward={r.get('episode_reward'):.2f} "
          f"killed_value_units={sc.get('killed_value_units')} score={sc.get('score')} "
          f"final_step={r.get('step')}")
