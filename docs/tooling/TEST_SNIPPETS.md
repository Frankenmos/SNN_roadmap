# Test Snippets — quick-run templates

Small self-contained Python snippets used as manual smoke tests during
the Fix-2 and Fix-1 work in this session. Not a substitute for real
tests in `tests/` — just saved here so you (or future me) can reuse
them when touching the same machinery. Each snippet assumes the conda
env is `sc2_ppo` and you run from the repo root.

---

## 1. LR scheduler wiring (cosine decay)

Builds a minimal fake net, constructs `PPO` with `total_updates`, and
checks the LR actually decays under the *correct* step order
(`optimizer.step` before `scheduler.step`).

```python
import torch, torch.nn as nn
class FakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.device = torch.device('cpu')
        self.use_amp = False
        self.amp_dtype = torch.float32
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)

from PPO_CNN.PPO import PPO
n = FakeNet()
p = PPO(n, lr=1e-4, total_updates=1000, lr_min=1e-5)

samples = [0, 100, 500, 900, 999]
lrs = {}
for i in range(1000):
    loss = n.fc(torch.randn(1, 4)).sum()
    p.optimizer.zero_grad(); loss.backward(); p.optimizer.step()
    if i in samples:
        lrs[i] = p.optimizer.param_groups[0]['lr']
    p.scheduler.step()
for k in sorted(lrs):
    print(f'update {k:4d}: lr={lrs[k]:.3e}')
```

Expected: LR starts at 1e-4, ends near 1e-5, curve is a half-cosine.

---

## 2. `_calculate_losses` shape + normalized-entropy check

Feeds random logits through the screen-point loss fn, verifies shapes
and that `entropy_mean` lives on a sane scale after normalization.

```python
import torch, torch.nn as nn
class FakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.device = torch.device('cpu')
        self.use_amp = False
        self.amp_dtype = torch.float32
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)

from PPO_CNN.PPO import PPO
p = PPO(FakeNet(), lr=1e-4)

B = 128
pl, vl, el, diag = p._calculate_losses(
    torch.randn(B, 3),
    torch.randn(B, 84),
    torch.randn(B, 84),
    torch.randn(B),              # state_values
    torch.randint(0, 3, (B,)),   # actions
    torch.randint(0, 84, (B,)),  # move_x
    torch.randint(0, 84, (B,)),  # move_y
    torch.randn(B),              # old_log_probs
    torch.randn(B),              # advantages
    torch.randn(B),              # returns
)
print('policy_loss  ', float(pl))
print('value_loss   ', float(vl))
print('entropy_loss ', float(el))
print('entropy_mean ', float(diag['entropy_mean']))
print('approx_kl    ', float(diag['approx_kl']))
print('clip_frac    ', float(diag['clip_frac']))
```

Sanity: after the Fix-2 normalization, `entropy_mean` should sit in
roughly `[1.0, 3.0]` (attack contributes 1, move up to 3 because three
normalized heads). Without the fix it would be ~9+ for move-heavy
batches.

---

## 3. State-replay end-to-end (Fix 1)

Builds a real `PolicyNetwork` on a reduced input size, stores a small
batch of transitions with their pre-step states, runs one
`update_policy` and checks that (a) the stats dict is sane, (b) params
actually moved, (c) memory is cleared.

```python
import torch
from PPO_CNN.policy_network import PolicyNetwork
from PPO_CNN.PPO import PPO

torch.manual_seed(0)
net = PolicyNetwork((3, 16, 16), vector_input_dim=8, action_dim=3,
                    num_steps=2)
net.device = torch.device('cpu'); net.to('cpu')
net.use_amp = False; net.amp_dtype = torch.float32

ppo = PPO(net, lr=1e-4, total_updates=0, lr_min=0.0)

T = 6
for i in range(T):
    state = net.init_concrete_state(batch_size=1)
    spatial = torch.randn(3, 16, 16)
    vector = torch.randn(8)
    with torch.no_grad():
        s, v = spatial.unsqueeze(0), vector.unsqueeze(0)
        _, _, _, sv, _ = net(s, v, state=state)
    ppo.store_transition(
        spatial, vector,
        torch.tensor(1), torch.tensor(5), torch.tensor(10),
        state,
        torch.tensor(-1.0), torch.tensor(1.0), torch.tensor(sv.item()),
        torch.tensor(i == T - 1, dtype=torch.float32),
    )

init = [p.clone() for p in net.parameters()]
losses, stats = ppo.update_policy(batch_size=3, epochs=1)
print(stats)
print('params changed:',
      sum(not torch.equal(a, b) for a, b in zip(init, net.parameters())),
      '/', len(init))
print('memory cleared:', len(ppo.memory) == 0)
```

Expected: prints a `[PPO] stacked SNN state: X GB` line. After the
SNN was moved from the conv stack to the post-attention token layer,
`X` should be in the tens-of-MB range (e.g. `~0.0001 GB` for this
reduced 6-transition smoke) instead of the multi-GB range the old
pixel-level LIF stack produced. The stats dict should include the PPO
diagnostics plus rollout/return stats, most parameter tensors should
change, and memory is cleared.

---

## 4. `get_friendly_health` helper (G1 verification)

Fake-obs smoke confirming the helper reads health from `feature_units`
and is independent of `obs.observation.player` contents. Also drives a
two-call `calculate_reward` sequence to prove the health-loss penalty
now actually fires.

```python
from types import SimpleNamespace
from obs_space.obs_space_2 import get_friendly_health
from PPO_CNN.reward_function_2 import RewardFunctionV2

def make_obs(friendly_health, enemy_health=100, enemy_count=1, last=False):
    friendly = SimpleNamespace(alliance=1, health=friendly_health,
                               x=0, y=0, unit_type=48, attack_range=5)
    enemies = [SimpleNamespace(alliance=4, health=enemy_health // enemy_count,
                               x=10, y=10, unit_type=110, attack_range=5)
               for _ in range(enemy_count)]
    feat = [friendly] + enemies
    return SimpleNamespace(
        observation=SimpleNamespace(
            # player_id constant: proves we're NOT reading these.
            player=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            feature_units=feat,
            score_cumulative=[0] * 13,
        ),
        last=lambda: last,
        reward=0,
    )

# Helper pure-read test.
assert get_friendly_health(make_obs(100)) == 100.0
assert get_friendly_health(make_obs(90)) == 90.0

# Reward-side: drop health by 10, expect health_reward == -4.
rf = RewardFunctionV2()
_ = rf.calculate_reward(make_obs(100), None)   # primes previous_*
r = rf.calculate_reward(make_obs(90), None)
comp = rf.get_last_reward_components()
print('total_reward =', r)
print('health_reward =', comp['health_reward'])  # expect -4.0
assert comp['health_reward'] == -4.0, comp
```

Expected: both asserts pass. Before G1 the health_reward would have
been 0.0 regardless of the health change.

---

## 5. Rollout-budget trainer trigger

Confirms the trainer now updates when the learnable-transition budget
is reached instead of on a fixed episode cadence.

```python
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import PPO_CNN_run as run_mod


class DummyReward:
    def calculate_reward(self, obs, _):
        return 1.0
    def get_last_reward_components(self):
        return None
    def reset(self):
        pass


class DummyPPO:
    def __init__(self):
        self.memory = []
        self.final_next = None
    def store_transition(self, *args, **kwargs):
        self.memory.append(1)
    def set_final_next(self, *args, **kwargs):
        self.final_next = True


class DummyPolicy:
    device = "cpu"


class DummyAgent:
    def __init__(self):
        self.policy = DummyPolicy()
        self.ppo = DummyPPO()
        self.reward_function = DummyReward()
        self._step = 0
        self.update_calls = 0
        self.snn_state = ("syn_next", "mem_next")
    def reset(self):
        self._step = 0
        self.reward_function.reset()
        self.snn_state = ("syn_next", "mem_next")
    def peek_observation(self, obs):
        return ("next_spatial", "next_vector")
    def step(self, obs):
        self._step += 1
        self.snn_state = ("syn_next", "mem_next")
        return (
            "noop", 0, 0, 0, ("syn", "mem"), 0.0, 0.0,
            "spatial", "vector", True,
        )
    def update_policy(self):
        self.update_calls += 1
        self.ppo.memory.clear()
        return {
            "mean_policy_loss": 0.0,
            "mean_value_loss": 0.0,
            "mean_entropy": 0.0,
            "mean_kl": 0.0,
            "clip_fraction": 0.0,
            "explained_variance": 1.0,
            "grad_norm": 0.0,
            "lr": 1e-4,
            "nonfinite_grad_steps": 0,
            "skipped_optimizer_steps": 0,
            "transitions_in_update": 4,
            "return_mean": 0.0,
            "return_std": 0.0,
            "return_p10": 0.0,
            "return_p50": 0.0,
            "return_p90": 0.0,
        }


class DummyObs:
    def __init__(self, max_steps, is_last=False):
        self.max_steps = max_steps
        self._last = is_last
        self.reward = 0.0
    def last(self):
        return self._last


class DummyEnv:
    def __init__(self, episode_lengths):
        self.episode_lengths = list(episode_lengths)
        self.episode_idx = -1
        self.step_in_episode = 0
        self.current_max = None
    def reset(self):
        self.episode_idx += 1
        self.current_max = self.episode_lengths[self.episode_idx]
        self.step_in_episode = 0
        return [DummyObs(self.current_max, is_last=False)]
    def step(self, actions):
        self.step_in_episode += 1
        return [
            DummyObs(
                self.current_max,
                is_last=self.step_in_episode >= self.current_max,
            )
        ]


class DummyQueue:
    def put(self, item):
        pass


env = DummyEnv([2, 2])
agent = DummyAgent()
queue = DummyQueue()

cfg_backup = {
    "total_episodes": run_mod.cfg.environment.total_episodes,
    "steps_per_episode": run_mod.cfg.environment.steps_per_episode,
    "reward_window": run_mod.cfg.environment.reward_window,
    "log_frequency": run_mod.cfg.environment.log_frequency,
    "rollout_steps": run_mod.cfg.hyperparameters.rollout_steps,
    "reward_scale": run_mod.cfg.hyperparameters.reward_scale,
    "eval_frequency": run_mod.cfg.environment.eval_frequency,
    "eval_episodes": run_mod.cfg.environment.eval_episodes,
}

run_mod.cfg.environment.total_episodes = 2
run_mod.cfg.environment.steps_per_episode = 10
run_mod.cfg.environment.reward_window = 10
run_mod.cfg.environment.log_frequency = 999
run_mod.cfg.environment.eval_frequency = 0
run_mod.cfg.environment.eval_episodes = 0
run_mod.cfg.hyperparameters.rollout_steps = 4
run_mod.cfg.hyperparameters.reward_scale = 1.0

try:
    with patch.object(run_mod, "load_checkpoint", return_value=(0, float("-inf"), deque(maxlen=10))):
        best = run_mod.train_agent(env, agent, None, queue)
    assert best == float("-inf")
    assert agent.update_calls == 1, agent.update_calls
finally:
    run_mod.cfg.environment.total_episodes = cfg_backup["total_episodes"]
    run_mod.cfg.environment.steps_per_episode = cfg_backup["steps_per_episode"]
    run_mod.cfg.environment.reward_window = cfg_backup["reward_window"]
    run_mod.cfg.environment.log_frequency = cfg_backup["log_frequency"]
    run_mod.cfg.environment.eval_frequency = cfg_backup["eval_frequency"]
    run_mod.cfg.environment.eval_episodes = cfg_backup["eval_episodes"]
    run_mod.cfg.hyperparameters.rollout_steps = cfg_backup["rollout_steps"]
    run_mod.cfg.hyperparameters.reward_scale = cfg_backup["reward_scale"]
```

Expected: the trainer performs exactly one PPO update after the rollout
budget of 4 learnable transitions is reached across two short episodes,
without relying on any fixed episode update cadence.

---

## Running

```powershell
# PowerShell one-liner pattern
& C:/Users/vladp/.conda/envs/sc2_ppo/python.exe -c "<paste snippet here>"
```

Or drop any snippet into a scratch `_tmp.py` and `python _tmp.py`.
