// Interactive snn.Synaptic neuron: exactly the update the policy's
// token-temporal pathways run per env step (agent_core/spiking_policy.py
// TokenTemporalSNN -> snnTorch Synaptic):
//   syn' = alpha·syn + I_t
//   mem' = beta·mem + syn' - reset_prev·theta      (theta = 1.0)
//   spk  = 1 if mem' > theta else 0                (Heaviside forward)
// Reset-by-subtraction lands on the NEXT step (snnTorch reset_delay).
import { useMemo, useState } from 'react'
import { pathwayTimeConstants } from '../data/runData'
import { Card, Eq, PresetButton, RepoRefs, Slider } from './controls.jsx'

const ACCENT = '#c084fc'
const T = 64
const THETA = 1.0

function simulate(alpha, beta, inputs, amplitude) {
  const syn = new Array(T)
  const mem = new Array(T)
  const spikes = new Array(T)
  let s = 0
  let m = 0
  let reset = 0
  for (let t = 0; t < T; t += 1) {
    s = alpha * s + inputs[t] * amplitude
    m = beta * m + s - reset * THETA
    const spk = m > THETA ? 1 : 0
    syn[t] = s
    mem[t] = m
    spikes[t] = spk
    reset = spk
  }
  return { syn, mem, spikes }
}

function linePath(values, width, height, min, max) {
  const span = max - min || 1
  return values
    .map(
      (value, index) =>
        `${index === 0 ? 'M' : 'L'}${((index / (values.length - 1)) * width).toFixed(1)},${(
          height - ((value - min) / span) * height
        ).toFixed(1)}`,
    )
    .join(' ')
}

const INPUT_PRESETS = {
  burst: () =>
    Array.from({ length: T }, (_, t) => (t >= 8 && t < 14 ? 1 : 0)),
  regular: () => Array.from({ length: T }, (_, t) => (t % 6 === 0 ? 1 : 0)),
  single: () => Array.from({ length: T }, (_, t) => (t === 10 ? 1 : 0)),
  dense: () =>
    Array.from({ length: T }, (_, t) => (t >= 6 && t % 2 === 0 ? 1 : 0)),
}

export function LifPlayground({ runData }) {
  const [alpha, setAlpha] = useState(0.55)
  const [beta, setBeta] = useState(0.65)
  const [amplitude, setAmplitude] = useState(0.55)
  const [inputs, setInputs] = useState(INPUT_PRESETS.regular)

  // Presets: config inits always; learned values when a bundle is loaded.
  const presets = useMemo(() => {
    const list = [
      { label: 'fast init 0.55 / 0.65', alpha: 0.55, beta: 0.65 },
      { label: 'slow init 0.92 / 0.97', alpha: 0.92, beta: 0.97 },
    ]
    const entry = runData?.entries[runData.entries.length - 1]
    if (entry) {
      const constants = pathwayTimeConstants(entry)
      for (const pathway of ['fast', 'slow']) {
        const a = constants[pathway].alpha?.effective_mean
        const b = constants[pathway].beta?.effective_mean
        if (a != null && b != null) {
          list.push({
            label: `${pathway} learned ${a.toFixed(2)} / ${b.toFixed(2)}`,
            alpha: a,
            beta: b,
            live: true,
          })
        }
      }
    }
    return list
  }, [runData])

  const { syn, mem, spikes } = useMemo(
    () => simulate(alpha, beta, inputs, amplitude),
    [alpha, beta, inputs, amplitude],
  )

  const width = 640
  const plotHeight = 120
  const memMax = Math.max(1.6, ...mem, ...syn)
  const memMin = Math.min(0, ...mem, ...syn)

  const toggleInput = (index) => {
    setInputs((current) => {
      const next = [...current]
      next[index] = next[index] ? 0 : 1
      return next
    })
  }

  return (
    <div className="space-y-4">
      <Card title="synaptic LIF neuron — the memory cell of both SNN pathways" accent={ACCENT}>
        <div className="grid gap-4 md:grid-cols-[240px,1fr]">
          <div className="space-y-3">
            <Slider label="α (synaptic current decay)" value={alpha} min={0} max={1}
              step={0.01} onChange={setAlpha} format={(v) => v.toFixed(2)} accent={ACCENT} />
            <Slider label="β (membrane decay)" value={beta} min={0} max={1}
              step={0.01} onChange={setBeta} format={(v) => v.toFixed(2)} accent={ACCENT} />
            <Slider label="input amplitude" value={amplitude} min={0.1} max={1.5}
              step={0.05} onChange={setAmplitude} format={(v) => v.toFixed(2)} accent={ACCENT} />
            <div className="flex flex-wrap gap-1.5">
              {presets.map((preset) => (
                <PresetButton
                  key={preset.label}
                  accent={preset.live ? '#34d399' : ACCENT}
                  active={
                    Math.abs(preset.alpha - alpha) < 0.005 &&
                    Math.abs(preset.beta - beta) < 0.005
                  }
                  onClick={() => {
                    setAlpha(preset.alpha)
                    setBeta(preset.beta)
                  }}
                >
                  {preset.label}
                </PresetButton>
              ))}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {Object.keys(INPUT_PRESETS).map((name) => (
                <PresetButton key={name} accent={ACCENT}
                  onClick={() => setInputs(INPUT_PRESETS[name]())}>
                  {name} input
                </PresetButton>
              ))}
            </div>
            <Eq>syn′ = α·syn + Iₜ</Eq>
            <Eq>mem′ = β·mem + syn′ − spk·θ, θ = 1</Eq>
          </div>

          <div className="min-w-0">
            {/* input spikes: paintable strip */}
            <div className="mb-1 font-mono text-[10px] text-slate-500">
              input spikes Iₜ — click cells to paint
            </div>
            <div className="mb-2 flex gap-[1px]">
              {inputs.map((value, index) => (
                <button
                  key={index}
                  onClick={() => toggleInput(index)}
                  className="h-5 flex-1 rounded-[2px] transition"
                  style={{
                    background: value ? '#fbbf24' : 'rgba(51,65,85,0.5)',
                  }}
                  title={`t=${index}`}
                />
              ))}
            </div>

            <svg
              viewBox={`0 0 ${width} ${plotHeight}`}
              className="w-full rounded-lg bg-[#070b18]"
              preserveAspectRatio="none"
              style={{ height: 150 }}
            >
              {/* threshold */}
              <line
                x1="0" x2={width}
                y1={plotHeight - ((THETA - memMin) / (memMax - memMin)) * plotHeight}
                y2={plotHeight - ((THETA - memMin) / (memMax - memMin)) * plotHeight}
                stroke="#f87171" strokeDasharray="4 4" strokeWidth="1" opacity="0.7"
              />
              <path d={linePath(syn, width, plotHeight, memMin, memMax)}
                fill="none" stroke="#fbbf24" strokeWidth="1.4" opacity="0.8" />
              <path d={linePath(mem, width, plotHeight, memMin, memMax)}
                fill="none" stroke={ACCENT} strokeWidth="1.8" />
            </svg>

            {/* output spike raster */}
            <div className="mt-2 flex gap-[1px]">
              {spikes.map((value, index) => (
                <div key={index} className="h-4 flex-1 rounded-[2px]"
                  style={{ background: value ? ACCENT : 'rgba(30,41,59,0.6)' }} />
              ))}
            </div>
            <div className="mt-2 flex gap-4 font-mono text-[10px] text-slate-500">
              <span><span className="text-amber-300">—</span> syn (current)</span>
              <span><span style={{ color: ACCENT }}>—</span> mem (membrane)</span>
              <span><span className="text-red-400">- -</span> θ = 1.0</span>
              <span style={{ color: ACCENT }}>▮ output spikes: {spikes.reduce((a, b) => a + b, 0)}</span>
            </div>
          </div>
        </div>
        <RepoRefs
          refs={[
            ['agent_core/spiking_policy.py', '316-345'],
            ['snnTorch Synaptic', 'threshold=1.0, reset_mechanism="subtract"'],
          ]}
          note={
            'One env step = one SNN step: state (syn, mem) leaves the network, rides ' +
            'through the environment, and comes back. Try slow-learned β=1.0: the ' +
            'membrane never leaks — v6 turned its slow pathway into a perfect integrator. ' +
            'Reset-by-subtraction lands one step late (snnTorch reset_delay).'
          }
        />
      </Card>
    </div>
  )
}
