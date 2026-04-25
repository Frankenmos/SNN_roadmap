# Observation Feedback Investigation - 2026-04-23

## Question

User asked: "How are actions passed back to the neural network? How are they encoded as policy input?"

## Investigation Results

### Current Action Feedback Loop

The agent DOES receive action feedback, but it's limited:

```
Action Taken → last_action_token [type_id, x, y, extra] → meta_vec[15:18] → Policy
```

**What the agent knows:**
- I clicked at (x, y)
- The action type (NO_OP, LEFT_CLICK, RIGHT_CLICK)

**What the agent does NOT know:**
- Did the action succeed or fail? (`action_result`)
- Am I under attack? (`alerts`)
- Game score progression (`score_cumulative`)

## Key Discovery: PySC2 Has Missing Fields

From [DeepMind's PySC2 documentation](https://github.com/deepmind/pysc2/blob/master/docs/environment.md):

### `action_result` (critical!)
A `(n)` tensor giving the result of the action. Values from `error.proto`.
This tells you if an action succeeded or failed!

### `alerts`
A `(n)` tensor (usually empty, max 2) for when you're being attacked.

### `score_cumulative`
A `(n)` tensor including:
- score
- idle_production_time
- idle_worker_time
- ... (more fields)

### `last_actions` (currently underutilized)
A `(n)` tensor of actions that succeeded since last obs.
Currently only the first element is used for `last_action_index`.

## Why This Matters

The agent may struggle to learn because:
1. **No success/failure signal** - Doesn't know if clicks registered
2. **No danger awareness** - Doesn't see `alerts`
3. **No progress signal** - Doesn't see score changes except through delayed reward

## Next Steps

See [observation_expansion_plan.md](observation_expansion_plan.md) for implementation plan.

## Sources

- [PySC2 Environment Documentation](https://github.com/deepmind/pysc2/blob/master/docs/environment.md) - Official DeepMind docs
