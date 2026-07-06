# SNN-PPO Architecture Explorer

An interactive 3D walkthrough of the real DefeatRoaches agent: the
inference pipeline laid out along a spline (PySC2 obs → extractor →
PolicyInputBatch → encoders → 95-token stream → spiking attention →
fast/slow token-temporal SNN → heads → dispatch) plus a toggleable
training-loop circuit (fragments → GAE → TBPTT → PPO → SIL) floating
above it.

Every tensor shape, formula, and code excerpt in the info panels was
pulled from this repository and cites its `file:line` source
(see `src/data/zones.js`, verified 2026-07-06). When code and docs
disagreed, the code won.

## Run

```bash
cd tools/viz/arch_explorer
npm install
npm run dev        # dev server, opens on http://localhost:5173
```

Production build + preview:

```bash
npm run build
npm run preview    # serves dist/ on http://localhost:4173
```

## Controls

| Input | Effect |
|-------|--------|
| Scroll wheel | Dolly the camera through the pipeline (input → output) |
| Hover a station | Neon highlight + tensor-shape tooltip |
| Click a station | Camera tween to the zone + slide-out info panel |
| `Esc` / click empty space | Close the panel, release the camera |
| "Training loop" button | Show/hide the elevated training circuit |

The info panel's ←/→ buttons walk the pipeline (or the training loop)
zone by zone.

## Live run data (optional)

The explorer can display the *learned* values of a real run next to the
static architecture story:

```bash
# from the repo root
python -m tools.registry export <run_name>
```

This writes `public/run_data.json` (gitignored): the run's snapshot /
checkpoint lineage with learned α/β time constants per artifact, the
rollout action mix, grad norms, SIL health, and eval scores joined from
`models/<run>/training_logs.db` (opened read-only).

With a bundle present:

- a **live run data** badge appears bottom-right;
- zones with a live slice (SNN pathways, Action Dispatch, Readout
  Heads, PPO, SIL) gain a **live** section in their info panel, with a
  dropdown to scrub across snapshots/checkpoints;
- the SNN station's pulse rings use the selected artifact's *learned*
  effective β for their decay envelopes — a slow pathway that learned
  β→1 (as v6 did) visibly stops decaying.

`npm run dev` picks the file up immediately; for `npm run preview`,
re-run `npm run build` (which copies `public/` into `dist/`). Snapshots
appear in the dropdown once snapshot recording is enabled in
`config.yaml` (`snapshot_*` keys under `distributed:`); without them
you scrub between `checkpoint` and `best`.

Raw α/β can drift outside [0, 1] during training; snnTorch clamps them
in the forward pass, so the panel shows the clamped *effective* value
(with the raw value alongside when they differ).

## Math Lab

The **∑ math lab** button (top right) opens interactive playgrounds for
the agent's core equations — every page cites its `file:line` source
and labels anything synthetic:

| Page | What you can do |
|------|-----------------|
| LIF neuron | Paint an input spike train, drive α/β sliders, watch (syn, mem) and output spikes; presets jump to the config inits and (with a live bundle) the run's *learned* constants |
| surrogate grad | The straight-through trick: hard step forward vs `1/(1+k\|U\|)²` backward, slope slider (repo default k=25) |
| GAE | Paint rewards over a 16-step fragment, γ/λ sliders, and the repo's truncation-vs-termination toggle — watch δ and A recompute via the exact backwards recursion |
| PPO clip | The clipped surrogate at the repo's ε=0.10, advantage slider, flat-region intuition |
| coarse→fine | The factorized 49×144 spatial click: place a hotspot, temperature sliders, sample a target through the exact `x = col_c·12 + col_f` composition |

`Esc` closes the lab back to the 3D scene.

## What the moving parts mean

- **Flow particles** trace tensors along the pipeline spline.
- **The token ring** (at the 95-Token Stream station) renders exactly
  95 instanced tokens color-coded by type: 49 spatial (cyan),
  24 entity (magenta), 20 selection (green), 1 action-feedback (amber),
  1 meta (white).
- **Spike flashes** (at Spiking Self-Attention) are discrete events with
  a sharp attack and fast decay — spikes, not glow.
- **The two orbit rings** (at Fast & Slow SNN) pulse at very different
  rates and decay envelopes: the fast pathway (α=0.55, β=0.65) flashes
  and forgets, the slow one (α=0.92, β=0.97) holds its charge — the
  dual-timescale story of the recurrent (syn, mem) state.

## Headless smoke test

```bash
npm run build
npm run smoke
```

Serves `dist/`, opens it in a headless system Edge/Chrome via
`puppeteer-core` (no browser download), waits for real rendered frames,
checks window-resize handling, and writes `smoke_screenshot.png`.
A second phase injects a synthetic `run_data.json` and asserts the live
sections render its learned constants and action mix
(`smoke_screenshot_live.png`).
Set `ARCH_EXPLORER_BROWSER=<path>` to point at a specific browser.

## Implementation notes

- React + React Three Fiber + drei + Tailwind via Vite; no external
  assets at runtime (labels are DOM overlays, the glow sprite is a
  generated canvas texture) so it works fully offline.
- All animated geometry is instanced; per-frame updates reuse
  module-scope temporaries (no per-frame allocation), and particle
  paths read from precomputed curve lookup tables.
