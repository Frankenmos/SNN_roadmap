// The PPO clipped surrogate with the repo's actual epsilon (0.10 -
// tighter than the common 0.2). L(r) = min(r·A, clip(r, 1-eps, 1+eps)·A)
// as a function of the probability ratio r = pi_new/pi_old.
import { useMemo, useState } from 'react'
import { Card, Eq, RepoRefs, Slider } from './controls.jsx'

const ACCENT = '#f59e0b'
const WIDTH = 640
const HEIGHT = 200
const R_MAX = 2
const SAMPLES = 201

function toX(r) {
  return (r / R_MAX) * WIDTH
}

export function ClipPlayground() {
  const [epsilon, setEpsilon] = useState(0.1)
  const [advantage, setAdvantage] = useState(1)

  const { clippedPath, rawPath, yZero } = useMemo(() => {
    const objective = (r) =>
      Math.min(
        r * advantage,
        Math.min(Math.max(r, 1 - epsilon), 1 + epsilon) * advantage,
      )
    const yMax = Math.max(Math.abs(advantage) * 1.6, 0.5)
    const yMin = -yMax
    const toY = (value) => HEIGHT - ((value - yMin) / (yMax - yMin)) * HEIGHT
    let clipped = ''
    let raw = ''
    for (let i = 0; i < SAMPLES; i += 1) {
      const r = (R_MAX * i) / (SAMPLES - 1)
      const command = i === 0 ? 'M' : 'L'
      clipped += `${command}${toX(r).toFixed(1)},${toY(objective(r)).toFixed(1)}`
      raw += `${command}${toX(r).toFixed(1)},${toY(r * advantage).toFixed(1)}`
    }
    return { clippedPath: clipped, rawPath: raw, yZero: toY(0) }
  }, [epsilon, advantage])

  return (
    <div className="space-y-4">
      <Card title="the trust region, drawn" accent={ACCENT}>
        <div className="grid gap-4 md:grid-cols-[240px,1fr]">
          <div className="space-y-3">
            <Slider label="ε (clip range)" value={epsilon} min={0.02} max={0.4}
              step={0.01} onChange={setEpsilon} format={(v) => v.toFixed(2)} accent={ACCENT} />
            <Slider label="advantage A" value={advantage} min={-2} max={2}
              step={0.1} onChange={setAdvantage} format={(v) => v.toFixed(1)} accent={ACCENT} />
            <Eq>r = π_new(a|s) / π_old(a|s)</Eq>
            <Eq>L = min(r·A, clip(r, 1−ε, 1+ε)·A)</Eq>
            <p className="font-mono text-[11px] leading-relaxed text-slate-400">
              Where the amber line goes flat, the gradient is zero: once the
              policy has moved the ratio past 1±ε in the direction the
              advantage rewards, this sample stops pushing. Flip A negative
              and the flat side flips too — the clip only ever *removes*
              incentive, never adds it.
            </p>
          </div>
          <div className="min-w-0">
            <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full rounded-lg bg-[#070b18]"
              preserveAspectRatio="none" style={{ height: 220 }}>
              <line x1="0" x2={WIDTH} y1={yZero} y2={yZero}
                stroke="#334155" strokeWidth="1" />
              {/* clip boundaries */}
              {[1 - epsilon, 1, 1 + epsilon].map((r, index) => (
                <line key={index} x1={toX(r)} x2={toX(r)} y1="0" y2={HEIGHT}
                  stroke={index === 1 ? '#475569' : '#f8717155'}
                  strokeWidth="1" strokeDasharray={index === 1 ? '2 6' : '4 4'} />
              ))}
              <path d={rawPath} fill="none" stroke="#94a3b8"
                strokeWidth="1.4" strokeDasharray="5 4" />
              <path d={clippedPath} fill="none" stroke={ACCENT} strokeWidth="2.2" />
            </svg>
            <div className="mt-1 flex gap-4 font-mono text-[10px] text-slate-500">
              <span><span style={{ color: ACCENT }}>—</span> clipped objective</span>
              <span><span className="text-slate-400">- -</span> unclipped r·A</span>
              <span><span className="text-red-400">|</span> 1−ε and 1+ε</span>
              <span>x: ratio r ∈ [0, 2]</span>
            </div>
          </div>
        </div>
        <RepoRefs
          refs={[
            ['agent_core/ppo_trainer.py', '2283-2288'],
            ['config.yaml', 'ppo.clip_epsilon: 0.10'],
          ]}
          note={
            'The repo runs ε=0.10 — half the textbook 0.2 — as a deliberate ' +
            'stability choice paired with target_kl early-stopping: with a ' +
            'spiking net and bf16, conservative clipping is the safer default ' +
            '(docs/current/AUDIT_RECONCILIATION_2026-06-29.md). clip_fraction ' +
            'in the dashboard is the share of samples sitting in the flat region.'
          }
        />
      </Card>
    </div>
  )
}
