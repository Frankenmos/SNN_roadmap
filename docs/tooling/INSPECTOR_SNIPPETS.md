# Observation Inspector — Script Snippets

Reusable one-shot Python probes for
`analysis_results/<run_name>/eval_observation_space.jsonl` dumps produced
by `PPO_CNN_eval.py --inspect`. Each snippet is self-contained; just
update the path and run via `python - <<'PY' ... PY`.

Keep this file as gifts to future sessions — when the obs schema
changes or a new run is dumped, these are the first things to reach for.

---

## 1. First-record structure dump

Quick look at one record. Good for seeing what keys the obs exposes and
a sample unit attribute view. **Warning**: single-record only — fields
that happen to be empty *in this record* will look empty even if they
are populated in other records. Use snippet #2 for truth.

```python
import json, sys
p = "analysis_results/<run_name>/eval_observation_space.jsonl"
with open(p) as f:
    r = json.loads(f.readline())
print("keys:", r["keys"])
print("available_actions_count:", r["available_actions_count"])
print("feature_units.count:", r["feature_units"]["count"])
print(
    "feature_units.sample[0]:",
    r["feature_units"]["sample"][0] if r["feature_units"]["sample"] else None,
)
```

---

## 2. Aggregate probe across all records

The real diagnostic. Tracks max first-dim shape across records,
per-field non-empty count, dtype, and overall value range. Distinguishes
"always empty" from "sometimes populated".

```python
import json
from collections import defaultdict

p = "analysis_results/<run_name>/eval_observation_space.jsonl"

max_dim0 = defaultdict(int)
ever_nonempty = defaultdict(int)
overall_min, overall_max, dtypes = {}, {}, {}

with open(p) as f:
    total = 0
    for line in f:
        total += 1
        r = json.loads(line)
        for k, v in r.get("fields", {}).items():
            if not isinstance(v, dict):
                continue
            if v.get("dtype"):
                dtypes[k] = v["dtype"]
            shape = v.get("shape")
            if shape and len(shape) >= 1:
                max_dim0[k] = max(max_dim0[k], shape[0])
                if shape[0] > 0:
                    ever_nonempty[k] += 1
            mn, mx = v.get("min"), v.get("max")
            if mn is not None:
                overall_min[k] = mn if k not in overall_min else min(overall_min[k], mn)
            if mx is not None:
                overall_max[k] = mx if k not in overall_max else max(overall_max[k], mx)

interesting = [
    "feature_units", "raw_units", "multi_select", "single_select",
    "build_queue", "cargo", "production_queue", "control_groups",
    "player", "last_actions", "available_actions", "score_cumulative",
    "score_by_category", "score_by_vital", "alerts", "action_result",
    "upgrades", "feature_minimap", "feature_effects", "radar",
    "raw_effects",
]
print(f"records: {total}")
print(f"{'field':22s} {'max_dim0':>8s} {'non-empty':>10s} {'dtype':>8s} {'min..max':>20s}")
for k in interesting:
    rng = f"{overall_min.get(k, '?')}..{overall_max.get(k, '?')}"
    print(
        f"{k:22s} {max_dim0.get(k, '-'):>8} "
        f"{ever_nonempty.get(k, 0):>10} {dtypes.get(k, '-'):>8s} {rng:>20s}"
    )
```

---

## 3. Unit-count distribution (for `N_max` sizing)

Answers "how big should `entity_features` be padded to?" — pull
percentiles from the real dump rather than guessing.

```python
import json, statistics

p = "analysis_results/<run_name>/eval_observation_space.jsonl"
counts = []
with open(p) as f:
    for line in f:
        r = json.loads(line)
        counts.append(r["feature_units"].get("count", 0))

counts.sort()
n = len(counts)
def pct(q): return counts[min(n - 1, int(n * q))]
print(
    f"records={n} min={counts[0]} max={counts[-1]} mean={statistics.mean(counts):.1f} "
    f"p50={pct(0.5)} p90={pct(0.9)} p95={pct(0.95)} p99={pct(0.99)}"
)
```

Use `p99` + small margin for `N_max`. Going above p99 wastes VRAM;
going below causes truncation during peak combat.

---

## 4. What's in `available_actions` actually

When designing action masking, the real PySC2 action IDs that show up
matter more than the theoretical list. This counts frequency per ID.

```python
import json
from collections import Counter

p = "analysis_results/<run_name>/eval_observation_space.jsonl"
freq = Counter()
# available_actions is logged only as a count in the summary today.
# If you want the full id set, extend ObservationInspectorWrapper to
# dump the list; then this snippet reads r["fields"]["available_actions"]
# or a new r["available_actions"] list field.
# Placeholder: read count distribution instead.
counts = []
with open(p) as f:
    for line in f:
        r = json.loads(line)
        if r.get("available_actions_count") is not None:
            counts.append(int(r["available_actions_count"]))

if counts:
    counts.sort()
    print(
        f"available_actions_count distribution: "
        f"min={counts[0]} max={counts[-1]} "
        f"p50={counts[len(counts)//2]} p99={counts[int(len(counts)*0.99)]}"
    )
```

**Note**: the wrapper currently only logs the *count*, not the ID list.
If we need per-ID frequency, extend `ObservationInspectorWrapper._log_timesteps`
to also dump `list(available_actions)` into the record.

---

## 5. Record cadence sanity-check

Are we sampling at the expected cadence? Useful when reading a dump and
wondering "did this come from one episode or many?".

```python
import json
from collections import Counter

p = "analysis_results/<run_name>/eval_observation_space.jsonl"
events = Counter()
episodes = set()
with open(p) as f:
    for line in f:
        r = json.loads(line)
        events[r["event"]] += 1
        episodes.add(r["episode"])

print(f"records_by_event: {dict(events)}")
print(f"episodes seen: {sorted(episodes)}")
```

If `events["reset"]` matches the number of episodes and `events["step"]`
is `(total_steps / log_every_n_steps)`-ish, the dump is complete.

---

## Suggested future extensions to `ObservationInspectorWrapper`

Not urgent, but the current summary throws away detail the probes could
use:

- Dump `list(available_actions)` per record, not just the count.
- Sample `feature_units` rows at full 46-attr width, not the 6-field
  curated subset in `_summarize_feature_units`.
- Dump `last_actions` value, not just the shape.
- Add `raw_units` samples with the `tag` field visible for identity
  tracking probes.

These are one-liner additions to `_log_timesteps` when the obs redesign
starts needing finer probes.
