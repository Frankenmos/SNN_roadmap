// The surrogate-gradient trick every spiking layer in the policy uses
// (agent_core/spiking_policy.py:490 -> surrogate.fast_sigmoid()).
// Forward: Heaviside step (a real spike). Backward: gradient of the
// fast sigmoid S(U) = U / (1 + k|U|):  dS/dU = 1 / (1 + k|U|)^2.
import { useMemo, useState } from 'react'
import { Card, Eq, RepoRefs, Slider } from './controls.jsx'

const ACCENT = '#f472b6'
const WIDTH = 640
const HEIGHT = 170
const U_RANGE = 2 // plot U - theta in [-2, 2]
const SAMPLES = 241

function path(fn, yMin, yMax) {
  const span = yMax - yMin || 1
  let d = ''
  for (let i = 0; i < SAMPLES; i += 1) {
    const u = -U_RANGE + (2 * U_RANGE * i) / (SAMPLES - 1)
    const x = ((u + U_RANGE) / (2 * U_RANGE)) * WIDTH
    const y = HEIGHT - ((fn(u) - yMin) / span) * HEIGHT
    d += `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }
  return d
}

export function SurrogatePlayground() {
  const [slope, setSlope] = useState(25)

  const paths = useMemo(() => {
    const heaviside = (u) => (u > 0 ? 1 : 0)
    const fastSigmoid = (u) => u / (1 + slope * Math.abs(u))
    const derivative = (u) => 1 / (1 + slope * Math.abs(u)) ** 2
    return {
      step: path(heaviside, -0.55, 1.1),
      approx: path(fastSigmoid, -0.55, 1.1),
      grad: path(derivative, 0, 1.05),
    }
  }, [slope])

  return (
    <div className="space-y-4">
      <Card title="why spikes can learn at all" accent={ACCENT}>
        <div className="grid gap-4 md:grid-cols-[240px,1fr]">
          <div className="space-y-3">
            <Slider label="slope k" value={slope} min={1} max={100} step={1}
              onChange={setSlope} format={(v) => v.toFixed(0)} accent={ACCENT} />
            <Eq>forward: spk = Θ(U − θ)</Eq>
            <Eq>backward: ∂spk/∂U ≈ 1 / (1 + k·|U−θ|)²</Eq>
            <p className="font-mono text-[11px] leading-relaxed text-slate-400">
              The step function's true derivative is 0 almost everywhere —
              backprop through a spike would learn nothing. The straight-through
              trick keeps the hard spike in the forward pass but pretends it was
              the smooth fast sigmoid on the way back. Larger k = tighter, more
              spike-like gradient window around the threshold.
            </p>
          </div>
          <div className="min-w-0 space-y-3">
            <div>
              <div className="mb-1 font-mono text-[10px] text-slate-500">
                forward: hard step (solid) vs fast-sigmoid approximation (dashed)
              </div>
              <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full rounded-lg bg-[#070b18]"
                preserveAspectRatio="none" style={{ height: 140 }}>
                <line x1={WIDTH / 2} x2={WIDTH / 2} y1="0" y2={HEIGHT}
                  stroke="#334155" strokeWidth="1" strokeDasharray="3 5" />
                <path d={paths.approx} fill="none" stroke="#94a3b8"
                  strokeWidth="1.4" strokeDasharray="5 4" />
                <path d={paths.step} fill="none" stroke={ACCENT} strokeWidth="2" />
              </svg>
            </div>
            <div>
              <div className="mb-1 font-mono text-[10px] text-slate-500">
                backward: the surrogate gradient 1/(1+k|U−θ|)²
              </div>
              <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full rounded-lg bg-[#070b18]"
                preserveAspectRatio="none" style={{ height: 140 }}>
                <line x1={WIDTH / 2} x2={WIDTH / 2} y1="0" y2={HEIGHT}
                  stroke="#334155" strokeWidth="1" strokeDasharray="3 5" />
                <path d={paths.grad} fill="none" stroke="#fbbf24" strokeWidth="2" />
              </svg>
              <div className="mt-1 font-mono text-[10px] text-slate-500">
                x axis: U − θ ∈ [−2, 2] · vertical line = the threshold
              </div>
            </div>
          </div>
        </div>
        <RepoRefs
          refs={[
            ['agent_core/spiking_policy.py', '490'],
            ['snnTorch surrogate.fast_sigmoid', 'slope k = 25 (repo default)'],
          ]}
          note={
            'This one spike_grad instance is shared by the attention lif_q/k/v ' +
            'neurons and both token-temporal pathways — every gradient that ' +
            'reaches a weight in the SNN passed through this curve.'
          }
        />
      </Card>
    </div>
  )
}
