// Trace replay: scrub or play through a recorded eval episode - the
// real screen masks (friendly / enemy / selected from player_relative),
// the real clicks, and the value / reward streams, exactly as the agent
// experienced them. Data comes from trace_data.json
// (tools/analysis/trace_export.py); nothing here is synthetic.
import { useEffect, useMemo, useRef, useState } from 'react'
import { ACTION_LABELS, unpackMask } from '../data/traceData'

const SIZE = 84
const CLICK_TRAIL = 6

function formatNumber(value, digits = 3) {
  return value == null || !Number.isFinite(Number(value))
    ? 'n/a'
    : Number(value).toFixed(digits)
}

function Playhead({ values, color, height = 46, playIndex, label }) {
  const width = 260
  const finite = values.filter((v) => Number.isFinite(v))
  if (finite.length < 2) return null
  const min = Math.min(...finite)
  const max = Math.max(...finite)
  const span = max - min || 1
  const denom = Math.max(values.length - 1, 1)
  const points = values
    .map((value, index) =>
      Number.isFinite(value)
        ? `${((index / denom) * width).toFixed(1)},${(
            height - 3 - ((value - min) / span) * (height - 6)
          ).toFixed(1)}`
        : null,
    )
    .filter(Boolean)
    .join(' ')
  const x = (playIndex / denom) * width
  return (
    <div>
      <div className="mb-0.5 flex justify-between font-mono text-[10px] text-slate-500">
        <span>{label}</span>
        <span className="text-slate-300">
          {formatNumber(values[playIndex])}
        </span>
      </div>
      <svg width={width} height={height} className="block rounded bg-[#070b18]">
        <polyline points={points} fill="none" stroke={color} strokeWidth="1.3" opacity="0.9" />
        <line x1={x} x2={x} y1="0" y2={height} stroke="#f8fafc" strokeWidth="1" opacity="0.55" />
      </svg>
    </div>
  )
}

export function TraceReplay({ trace, onClose }) {
  const steps = trace.steps
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(4) // steps per second
  const [showGrid, setShowGrid] = useState(true)
  const canvasRef = useRef(null)
  const stripRef = useRef(null)

  const step = steps[index]

  // Decoded masks for the current step (cached per index).
  const masks = useMemo(() => {
    if (!step?.friendly) return null
    return {
      friendly: unpackMask(step.friendly, SIZE),
      enemy: unpackMask(step.enemy, SIZE),
      selected: unpackMask(step.selected, SIZE),
    }
  }, [step])

  const values = useMemo(() => steps.map((s) => s.value), [steps])
  const cumRewards = useMemo(() => steps.map((s) => s.cum_reward), [steps])

  // Playback clock.
  useEffect(() => {
    if (!playing) return undefined
    const timer = setInterval(() => {
      setIndex((current) => {
        if (current >= steps.length - 1) {
          setPlaying(false)
          return current
        }
        return current + 1
      })
    }, 1000 / speed)
    return () => clearInterval(timer)
  }, [playing, speed, steps.length])

  // Screen canvas: friendly cyan, enemy red, selected brightened.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    const image = context.createImageData(SIZE, SIZE)
    for (let i = 0; i < SIZE * SIZE; i += 1) {
      image.data[i * 4] = 7
      image.data[i * 4 + 1] = 11
      image.data[i * 4 + 2] = 24
      image.data[i * 4 + 3] = 255
    }
    if (masks) {
      for (let i = 0; i < SIZE * SIZE; i += 1) {
        if (masks.friendly[i]) {
          const bright = masks.selected[i] ? 1 : 0.62
          image.data[i * 4] = 34 * bright + (masks.selected[i] ? 120 : 0)
          image.data[i * 4 + 1] = 211 * bright
          image.data[i * 4 + 2] = 238 * bright
        } else if (masks.enemy[i]) {
          image.data[i * 4] = 248
          image.data[i * 4 + 1] = 90
          image.data[i * 4 + 2] = 90
        }
      }
    }
    context.putImageData(image, 0, 0)
  }, [masks])

  // Action strip: one pixel column per step, drawn once.
  useEffect(() => {
    const canvas = stripRef.current
    if (!canvas) return
    const context = canvas.getContext('2d')
    const colors = { 0: '#475569', 1: '#f59e0b', 2: '#22d3ee' }
    context.fillStyle = '#0b1122'
    context.fillRect(0, 0, steps.length, 1)
    steps.forEach((s, i) => {
      context.fillStyle = s.policy ? colors[s.action] ?? '#1e293b' : '#1e293b'
      context.fillRect(i, 0, 1, 1)
    })
  }, [steps])

  // Click trail: recent right-clicks up to the playhead.
  const clicks = useMemo(() => {
    const list = []
    for (let i = index; i >= 0 && list.length < CLICK_TRAIL; i -= 1) {
      const s = steps[i]
      if (s.policy && s.action === 2) list.push({ x: s.x, y: s.y, age: index - i })
    }
    return list
  }, [steps, index])

  const seekFromStrip = (event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const fraction = (event.clientX - rect.left) / rect.width
    setIndex(
      Math.min(
        steps.length - 1,
        Math.max(0, Math.round(fraction * (steps.length - 1))),
      ),
    )
  }

  const feedbackActive = (step.feedback ?? []).filter((v) => Math.abs(v) > 1e-3).length

  return (
    <div className="absolute inset-0 z-40 flex flex-col bg-[#050510]/92 backdrop-blur-sm">
      <header className="flex items-start justify-between px-6 pb-2 pt-5">
        <div>
          <h1 className="font-mono text-sm font-bold uppercase tracking-[0.3em] text-cyan-300">
            Trace Replay
          </h1>
          <p className="mt-1 font-mono text-[10px] text-slate-500">
            {trace.run} · episode {trace.episode_index} ({trace.mode}) ·{' '}
            {trace.steps_total} steps · total reward{' '}
            {formatNumber(trace.total_reward, 1)}
            {trace.checkpoint_episode != null &&
              ` · checkpoint @ episode ${trace.checkpoint_episode}`}
          </p>
        </div>
        <button
          onClick={onClose}
          className="glass rounded-2xl px-4 py-2 font-mono text-xs uppercase tracking-widest text-slate-300 transition hover:text-white"
          title="Back to the 3D scene (Esc)"
        >
          ← 3D scene
        </button>
      </header>

      <div className="flex min-h-0 flex-1 flex-wrap content-start gap-5 overflow-y-auto px-6 pb-6 pt-2 neon-scroll">
        {/* screen view */}
        <div className="glass rounded-2xl p-4">
          <div className="mb-2 flex items-center justify-between gap-6 font-mono text-[10px] text-slate-500">
            <span>feature_screen · player_relative decoded</span>
            <button
              onClick={() => setShowGrid((v) => !v)}
              className="text-slate-400 transition hover:text-slate-200"
            >
              {showGrid ? '◉' : '○'} 7×7 coarse grid
            </button>
          </div>
          <div className="relative" style={{ width: 336, height: 336 }}>
            <canvas
              ref={canvasRef}
              width={SIZE}
              height={SIZE}
              className="h-full w-full rounded-lg"
              style={{ imageRendering: 'pixelated' }}
            />
            {showGrid &&
              Array.from({ length: 6 }, (_, i) => (
                <div key={i}>
                  <div className="pointer-events-none absolute top-0 h-full w-px bg-cyan-300/10"
                    style={{ left: `${((i + 1) / 7) * 100}%` }} />
                  <div className="pointer-events-none absolute left-0 h-px w-full bg-cyan-300/10"
                    style={{ top: `${((i + 1) / 7) * 100}%` }} />
                </div>
              ))}
            {clicks.map((click) => (
              <div
                key={`${click.x}-${click.y}-${click.age}`}
                className="pointer-events-none absolute h-3 w-3 -translate-x-1/2 -translate-y-1/2"
                style={{
                  left: `${(click.x / SIZE) * 100}%`,
                  top: `${(click.y / SIZE) * 100}%`,
                  opacity: 1 - click.age / (CLICK_TRAIL + 2),
                }}
              >
                <div className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-amber-300" />
                <div className="absolute left-0 top-1/2 h-px w-full -translate-y-1/2 bg-amber-300" />
              </div>
            ))}
          </div>
          <div className="mt-2 flex gap-4 font-mono text-[10px] text-slate-500">
            <span className="text-cyan-300">■ friendly</span>
            <span className="text-red-400">■ enemy</span>
            <span className="text-slate-200">■ selected</span>
            <span className="text-amber-300">+ click</span>
          </div>
        </div>

        {/* transport + readouts */}
        <div className="flex min-w-[300px] max-w-md flex-1 flex-col gap-3">
          <div className="glass rounded-2xl p-4">
            <div className="mb-2 flex items-center gap-2">
              <button
                onClick={() => setPlaying((v) => !v)}
                className="rounded-lg border border-cyan-400/40 bg-cyan-400/10 px-4 py-1.5 font-mono text-xs text-cyan-200 transition hover:brightness-125"
              >
                {playing ? '❚❚ pause' : '▶ play'}
              </button>
              {[2, 4, 8].map((rate) => (
                <button key={rate} onClick={() => setSpeed(rate)}
                  className={`rounded-lg border px-2 py-1.5 font-mono text-[10px] transition
                              ${speed === rate ? 'border-cyan-400/50 text-cyan-200' : 'border-slate-600/40 text-slate-400 hover:text-slate-200'}`}>
                  {rate}×
                </button>
              ))}
              <span className="ml-auto font-mono text-[11px] text-slate-300">
                step {step.t} / {steps[steps.length - 1].t}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={steps.length - 1}
              value={index}
              onChange={(event) => setIndex(Number(event.target.value))}
              className="w-full"
              style={{ accentColor: '#22d3ee' }}
            />
            <div className="relative mt-1 h-2 w-full cursor-pointer overflow-hidden rounded"
              onClick={seekFromStrip} title="action per step - click to seek">
              <canvas ref={stripRef} width={steps.length} height={1}
                className="h-full w-full" style={{ imageRendering: 'pixelated' }} />
              <div className="pointer-events-none absolute top-0 h-full w-px bg-white/80"
                style={{ left: `${(index / (steps.length - 1)) * 100}%` }} />
            </div>
            <div className="mt-1 flex gap-3 font-mono text-[9px] text-slate-500">
              <span className="text-slate-400">■ no_op</span>
              <span className="text-cyan-300">■ right_click</span>
              <span className="text-amber-300">■ left_click</span>
              <span>■ non-policy</span>
            </div>
          </div>

          <div className="glass rounded-2xl p-4 font-mono text-[11px]">
            <table className="w-full text-left">
              <tbody>
                <tr>
                  <td className="text-slate-500">action</td>
                  <td className="text-right text-slate-100">
                    {step.policy ? ACTION_LABELS[step.action] ?? step.action : 'bootstrap (outside PPO)'}
                    {step.policy && step.action === 2 && ` → (${step.x}, ${step.y})`}
                  </td>
                </tr>
                <tr>
                  <td className="text-slate-500">dispatched</td>
                  <td className="text-right text-slate-200">{step.func ?? 'n/a'}</td>
                </tr>
                <tr>
                  <td className="text-slate-500">log π(a|s)</td>
                  <td className="text-right text-slate-200">{formatNumber(step.log_prob)}</td>
                </tr>
                <tr>
                  <td className="text-slate-500">V(s)</td>
                  <td className="text-right text-slate-200">{formatNumber(step.value)}</td>
                </tr>
                <tr>
                  <td className="text-slate-500">reward / cumulative</td>
                  <td className="text-right text-slate-200">
                    {formatNumber(step.reward, 2)} / {formatNumber(step.cum_reward, 2)}
                  </td>
                </tr>
                <tr>
                  <td className="text-slate-500">entities / selected</td>
                  <td className="text-right text-slate-200">
                    {step.entities ?? 'n/a'} / {step.selection ?? 'n/a'}
                  </td>
                </tr>
                <tr>
                  <td className="text-slate-500">feedback dims active</td>
                  <td className="text-right text-slate-200">{feedbackActive} / 12</td>
                </tr>
                <tr>
                  <td className="text-slate-500">learnable</td>
                  <td className="text-right">
                    <span className={step.learnable ? 'text-emerald-300' : 'text-slate-400'}>
                      {step.learnable ? 'yes' : 'no (masked from PPO)'}
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="glass flex flex-wrap gap-4 rounded-2xl p-4">
            <Playhead values={values} color="#c084fc" playIndex={index}
              label="V(s) — the critic's running guess" />
            <Playhead values={cumRewards} color="#4ade80" playIndex={index}
              label="cumulative shaped reward" />
          </div>
        </div>
      </div>
    </div>
  )
}
