# TinySkirmish Protocol Lab

Legible skirmish environment for testing StarCraft-shaped policy inputs
without launching SC2. The base environment is numpy-only; the bridge
modules import Torch and the real `agent_core` architecture directly.

Originally authored as a standalone Codex experiment (`E:\SNN\codex-experiment`,
2026-06-28); imported into this repo 2026-07-02 with the vendored
`policy_protocol` copy removed in favor of plain `agent_core` imports.

Every reward is an itemized dict of named parts validated to sum to the
total — reward attribution is inspectable per step, which is the point:
this env exists to verify the architecture and credit assignment on a
world small enough to reason about.

## Run (from the repo root)

```powershell
python -m envs.tiny_skirmish.self_check
python scripts\run_tiny_skirmish.py --mode scripted --episodes 1 --seed 9 --max-steps 40
python scripts\run_tiny_skirmish.py --mode random --episodes 2 --seed 2 --max-steps 20
python -m envs.tiny_skirmish.torch_self_check
python -m envs.tiny_skirmish.real_snn_bridge --device cpu --small
python -m envs.tiny_skirmish.real_snn_rollout --device cpu --small --max-steps 8
python -m envs.tiny_skirmish.render --mode scripted --seed 9 --steps 12 --out analysis_results\tiny_skirmish\scripted
python -m envs.tiny_skirmish.render --mode scripted --channels all --out analysis_results\tiny_skirmish\all_channels
python -m envs.tiny_skirmish.render_self_check
python -m envs.tiny_skirmish.live --mode scripted --seed 9 --fps 4
python -m envs.tiny_skirmish.live --mode manual --seed 9
python -m envs.tiny_skirmish.live_self_check
```

The self-checks also run under pytest via `tests/test_tiny_skirmish.py`
(render/live checks skip if Pillow/pygame are unavailable; pygame is not a
repo dependency — install it only if you want the live dashboard).

## Protocol Shape

- `spatial_obs`: `27 x 84 x 84`
- `entity_features`: `24 x 21`
- `selection_features`: `20 x 7`
- `action_feedback_tokens`: `1 x 12`
- `meta_vec`: `15`

Actions are semantic: `NO_OP`, placeholder `LEFT_CLICK`, and
`RIGHT_CLICK(x, y)` as the main spatial command.

The shapes mirror `agent_core.policy_protocol` (protocol v3) exactly and are
imported from it, so protocol drift breaks these modules loudly. NOTE: the 27
spatial channels are hand-authored (see below), NOT the PySC2 `feature_screen`
layer semantics — this is a twin of the tensor interface, not of StarCraft.

`real_snn_bridge` runs a real `PolicyNetwork` forward pass on a TinySkirmish
observation. `real_snn_rollout` goes one step further and collects a real PPO
`RolloutFragment` using the actual `PolicyNetwork` and `PPO` classes.

## Channel Renderer

The PNG renderer writes a world overview plus a spatial-channel contact sheet.
Core channels are:

- `0 walls`
- `1 friendly`
- `2 enemy`
- `3 selected`
- `4 friendly_hp`
- `5 enemy_hp`
- `6 passable`
- `7 friendly_attack_range`
- `8 last_target`
- `10 enemy_attack_range`
- `11 inverse_enemy_distance`
- `12 inverse_selected_distance`
- `13 bias`

## Live Renderer

`python -m envs.tiny_skirmish.live` opens a Pygame dashboard over the same
TinySkirmish protocol. It shows the live board, selected spatial channels,
last action, reward parts, events, and termination state.

Useful controls:

- `Space`: pause or resume autoplay
- `N`: step once while paused
- `R`: reset with the same seed
- `S` / `D` / `M`: scripted, random, or manual mode
- `C`: toggle core/all channels
- `Tab`: page through all-channel tiles
- `+` / `-`: adjust autoplay FPS
- mouse clicks in manual mode: left click placeholder or right click target
- `Esc`: quit
