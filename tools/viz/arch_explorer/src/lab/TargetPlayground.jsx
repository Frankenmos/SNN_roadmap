// The factorized spatial click: an 84x84 target is sampled as a 49-way
// coarse cell TIMES a 144-way fine offset inside it, composed exactly
// like agent_core/target_heads.py:738-742:
//   x = coarse_col·12 + fine_col,  y = coarse_row·12 + fine_row
// Distributions here are synthetic softmaxes around a hotspot you place;
// the mechanics (grids, composition, factorized log-prob) are the real ones.
import { useEffect, useMemo, useRef, useState } from 'react'
import { Card, Eq, RepoRefs, Slider } from './controls.jsx'

const ACCENT = '#22d3ee'
const COARSE = 7
const LOCAL = 12
const SCREEN = COARSE * LOCAL // 84

function softmax(scores) {
  const max = Math.max(...scores)
  const exps = scores.map((s) => Math.exp(s - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / sum)
}

// p_coarse over 49 cells: softmax of -dist(cell center, hotspot)/tau
function coarseDistribution(hotspot, tau) {
  const scores = []
  for (let row = 0; row < COARSE; row += 1) {
    for (let col = 0; col < COARSE; col += 1) {
      const cx = col * LOCAL + LOCAL / 2
      const cy = row * LOCAL + LOCAL / 2
      const dist = Math.hypot(cx - hotspot.x, cy - hotspot.y)
      scores.push(-dist / tau)
    }
  }
  return softmax(scores)
}

// p_fine over 144 offsets, conditioned on a coarse cell.
function fineDistribution(cellRow, cellCol, hotspot, tau) {
  const scores = []
  for (let fr = 0; fr < LOCAL; fr += 1) {
    for (let fc = 0; fc < LOCAL; fc += 1) {
      const x = cellCol * LOCAL + fc
      const y = cellRow * LOCAL + fr
      scores.push(-Math.hypot(x - hotspot.x, y - hotspot.y) / tau)
    }
  }
  return softmax(scores)
}

function sampleIndex(probabilities) {
  let u = Math.random()
  for (let i = 0; i < probabilities.length; i += 1) {
    u -= probabilities[i]
    if (u <= 0) return i
  }
  return probabilities.length - 1
}

export function TargetPlayground() {
  const [hotspot, setHotspot] = useState({ x: 52, y: 30 })
  const [coarseTau, setCoarseTau] = useState(12)
  const [fineTau, setFineTau] = useState(6)
  const [picked, setPicked] = useState(null) // {coarseIndex, fineIndex}
  const canvasRef = useRef(null)

  const pCoarse = useMemo(
    () => coarseDistribution(hotspot, coarseTau),
    [hotspot, coarseTau],
  )
  const selectedCoarse =
    picked?.coarseIndex ?? pCoarse.indexOf(Math.max(...pCoarse))
  const cellRow = Math.floor(selectedCoarse / COARSE)
  const cellCol = selectedCoarse % COARSE
  const pFine = useMemo(
    () => fineDistribution(cellRow, cellCol, hotspot, fineTau),
    [cellRow, cellCol, hotspot, fineTau],
  )

  const target = useMemo(() => {
    if (picked?.fineIndex == null) return null
    // exact composition from target_heads.py:738-742
    const fineRow = Math.floor(picked.fineIndex / LOCAL)
    const fineCol = picked.fineIndex % LOCAL
    return {
      x: cellCol * LOCAL + fineCol,
      y: cellRow * LOCAL + fineRow,
      logp:
        Math.log(pCoarse[selectedCoarse]) + Math.log(pFine[picked.fineIndex]),
    }
  }, [picked, cellRow, cellCol, pCoarse, pFine, selectedCoarse])

  // Composed 84x84 heatmap: p(x, y) = p_coarse(cell) * p_fine(offset|cell)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    const image = context.createImageData(SCREEN, SCREEN)
    let maxP = 0
    const composed = new Float32Array(SCREEN * SCREEN)
    for (let row = 0; row < COARSE; row += 1) {
      for (let col = 0; col < COARSE; col += 1) {
        const pc = pCoarse[row * COARSE + col]
        const fine = fineDistribution(row, col, hotspot, fineTau)
        for (let fr = 0; fr < LOCAL; fr += 1) {
          for (let fc = 0; fc < LOCAL; fc += 1) {
            const p = pc * fine[fr * LOCAL + fc]
            const index = (row * LOCAL + fr) * SCREEN + (col * LOCAL + fc)
            composed[index] = p
            if (p > maxP) maxP = p
          }
        }
      }
    }
    for (let i = 0; i < composed.length; i += 1) {
      const v = composed[i] / (maxP || 1)
      image.data[i * 4] = 8 + v * 26
      image.data[i * 4 + 1] = 12 + v * 199
      image.data[i * 4 + 2] = 26 + v * 212
      image.data[i * 4 + 3] = 255
    }
    context.putImageData(image, 0, 0)
  }, [pCoarse, hotspot, fineTau])

  const placeHotspot = (event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = Math.floor(((event.clientX - rect.left) / rect.width) * SCREEN)
    const y = Math.floor(((event.clientY - rect.top) / rect.height) * SCREEN)
    setHotspot({
      x: Math.min(SCREEN - 1, Math.max(0, x)),
      y: Math.min(SCREEN - 1, Math.max(0, y)),
    })
    setPicked(null)
  }

  const sample = () => {
    const coarseIndex = sampleIndex(pCoarse)
    const row = Math.floor(coarseIndex / COARSE)
    const col = coarseIndex % COARSE
    const fineIndex = sampleIndex(fineDistribution(row, col, hotspot, fineTau))
    setPicked({ coarseIndex, fineIndex })
  }

  const cellPercent = 100 / COARSE

  return (
    <div className="space-y-4">
      <Card title="49-way coarse × 144-way fine = one 84×84 click" accent={ACCENT}>
        <div className="grid gap-4 md:grid-cols-[240px,1fr]">
          <div className="space-y-3">
            <Slider label="coarse temperature τ_c" value={coarseTau} min={3} max={40}
              step={1} onChange={setCoarseTau} format={(v) => v.toFixed(0)} accent={ACCENT} />
            <Slider label="fine temperature τ_f" value={fineTau} min={1} max={20}
              step={0.5} onChange={setFineTau} format={(v) => v.toFixed(1)} accent={ACCENT} />
            <button
              onClick={sample}
              className="w-full rounded-lg border px-3 py-2 font-mono text-[11px] text-slate-100 transition hover:brightness-125"
              style={{ borderColor: `${ACCENT}66`, background: `${ACCENT}14` }}
            >
              sample coarse → fine → (x, y)
            </button>
            <Eq>p(x, y) = p_coarse(cell) · p_fine(offset | cell)</Eq>
            <Eq>x = col_c·12 + col_f&nbsp;&nbsp;y = row_c·12 + row_f</Eq>
            {target && (
              <div className="rounded-lg border border-slate-700/50 bg-slate-900/60 p-2 font-mono text-[11px] text-slate-300">
                coarse ({cellRow}, {cellCol}) · fine (
                {Math.floor(picked.fineIndex / LOCAL)}, {picked.fineIndex % LOCAL})
                <br />
                → pixel ({target.x}, {target.y})
                <br />
                log p = {target.logp.toFixed(3)}
              </div>
            )}
            <p className="font-mono text-[11px] leading-relaxed text-slate-400">
              One flat 7 056-way softmax would have to learn every pixel
              independently. Factorizing turns it into 49 + 144 choices, and
              the fine stage can specialize per cell — click the map to move
              the hotspot and watch both stages re-aim.
            </p>
          </div>
          <div className="min-w-0">
            <div className="mb-1 font-mono text-[10px] text-slate-500">
              composed distribution over the 84×84 screen — click to place the
              hotspot · grid = coarse cells
            </div>
            <div
              className="relative w-full cursor-crosshair overflow-hidden rounded-lg"
              style={{ aspectRatio: '1 / 1', maxWidth: 420 }}
              onClick={placeHotspot}
            >
              <canvas
                ref={canvasRef}
                width={SCREEN}
                height={SCREEN}
                className="h-full w-full"
                style={{ imageRendering: 'pixelated' }}
              />
              {/* coarse grid lines */}
              {Array.from({ length: COARSE - 1 }, (_, i) => (
                <div key={`v${i}`} className="pointer-events-none absolute top-0 h-full w-px bg-cyan-300/15"
                  style={{ left: `${(i + 1) * cellPercent}%` }} />
              ))}
              {Array.from({ length: COARSE - 1 }, (_, i) => (
                <div key={`h${i}`} className="pointer-events-none absolute left-0 h-px w-full bg-cyan-300/15"
                  style={{ top: `${(i + 1) * cellPercent}%` }} />
              ))}
              {/* selected coarse cell */}
              <div
                className="pointer-events-none absolute border"
                style={{
                  borderColor: `${ACCENT}aa`,
                  left: `${cellCol * cellPercent}%`,
                  top: `${cellRow * cellPercent}%`,
                  width: `${cellPercent}%`,
                  height: `${cellPercent}%`,
                }}
              />
              {/* hotspot + sampled pixel */}
              <div className="pointer-events-none absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-amber-300"
                style={{ left: `${(hotspot.x / SCREEN) * 100}%`, top: `${(hotspot.y / SCREEN) * 100}%` }} />
              {target && (
                <div className="pointer-events-none absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2"
                  style={{ left: `${(target.x / SCREEN) * 100}%`, top: `${(target.y / SCREEN) * 100}%` }}>
                  <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-red-400" />
                  <div className="absolute left-0 top-1/2 h-px w-full -translate-y-1/2 bg-red-400" />
                </div>
              )}
            </div>
          </div>
        </div>
        <RepoRefs
          refs={[
            ['agent_core/target_heads.py', '417-548 (head), 738-742 (composition)'],
            ['docs/current/ARCHITECTURE.md', 'spatial_head_type: coarse_to_fine'],
          ]}
          note={
            'In the real head the coarse stage is an einsum over the 49 spatial ' +
            'tokens and the fine stage is an MLP (plus a pre-pool 84×84 skip ' +
            'added after V5 diagnostics caught a constant fine sub-index). The ' +
            'distributions here are synthetic stand-ins; grids, composition and ' +
            'factorized log-prob are exact.'
          }
        />
      </Card>
    </div>
  )
}
