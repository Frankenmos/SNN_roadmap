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
Set `ARCH_EXPLORER_BROWSER=<path>` to point at a specific browser.

## Implementation notes

- React + React Three Fiber + drei + Tailwind via Vite; no external
  assets at runtime (labels are DOM overlays, the glow sprite is a
  generated canvas texture) so it works fully offline.
- All animated geometry is instanced; per-frame updates reuse
  module-scope temporaries (no per-frame allocation), and particle
  paths read from precomputed curve lookup tables.
