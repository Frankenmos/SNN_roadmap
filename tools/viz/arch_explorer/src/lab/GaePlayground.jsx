// GAE exactly as agent_core/ppo_trainer.py:_compute_advantages runs it
// (lines 2223-2247), including the repo's truncation-vs-termination
// distinction: a time cap stores done=0 and bootstraps from V(s_next);
// a real terminal stores done=1 and the bootstrap is masked off.
import { useMemo, useState } from 'react'
import { Card, Eq, PresetButton, RepoRefs, Slider } from './controls.jsx'

const ACCENT = '#fbbf24'
const T = 16
const REWARD_CYCLE = [0, 1, 2, -1]

// Mirrors the backwards recursion verbatim.
function computeGae({ rewards, values, gamma, lam, terminated, bootstrap }) {
  const dones = new Array(T).fill(0)
  if (terminated) dones[T - 1] = 1
  const deltas = new Array(T)
  const advantages = new Array(T)
  let running = 0
  let nextValue = bootstrap
  for (let t = T - 1; t >= 0; t -= 1) {
    const notDone = 1 - dones[t]
    const delta = rewards[t] + gamma * nextValue * notDone - values[t]
    running = delta + gamma * lam * notDone * running
    deltas[t] = delta
    advantages[t] = running
    nextValue = values[t]
  }
  return { deltas, advantages }
}

function Bars({ values, color, label }) {
  const max = Math.max(1, ...values.map((v) => Math.abs(v)))
  return (
    <div>
      <div className="mb-1 font-mono text-[10px] text-slate-500">{label}</div>
      <div className="flex h-20 items-center gap-[2px] rounded-lg bg-[#070b18] px-1">
        {values.map((value, index) => {
          const half = (Math.abs(value) / max) * 36
          return (
            <div key={index} className="relative h-full flex-1" title={`t=${index}: ${value.toFixed(3)}`}>
              <div className="absolute left-0 right-0 top-1/2 h-px bg-slate-700/60" />
              <div
                className="absolute left-[15%] right-[15%] rounded-[2px]"
                style={{
                  background: color,
                  opacity: 0.9,
                  height: `${half}px`,
                  top: value >= 0 ? `calc(50% - ${half}px)` : '50%',
                }}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function GaePlayground() {
  const [rewards, setRewards] = useState(() => {
    const initial = new Array(T).fill(0)
    initial[10] = 2 // a kill mid-fragment
    return initial
  })
  const [constantValue, setConstantValue] = useState(0.5)
  const [gamma, setGamma] = useState(0.99)
  const [lam, setLam] = useState(0.95)
  const [terminated, setTerminated] = useState(false)
  const [bootstrap, setBootstrap] = useState(0.5)

  const values = useMemo(() => new Array(T).fill(constantValue), [constantValue])
  const { deltas, advantages } = useMemo(
    () => computeGae({ rewards, values, gamma, lam, terminated, bootstrap }),
    [rewards, values, gamma, lam, terminated, bootstrap],
  )

  const cycleReward = (index) => {
    setRewards((current) => {
      const next = [...current]
      const position = REWARD_CYCLE.indexOf(next[index])
      next[index] = REWARD_CYCLE[(position + 1) % REWARD_CYCLE.length]
      return next
    })
  }

  return (
    <div className="space-y-4">
      <Card title="one fragment, sixteen steps" accent={ACCENT}>
        <div className="grid gap-4 md:grid-cols-[240px,1fr]">
          <div className="space-y-3">
            <Slider label="γ (discount)" value={gamma} min={0.8} max={1} step={0.005}
              onChange={setGamma} format={(v) => v.toFixed(3)} accent={ACCENT} />
            <Slider label="λ (GAE mixing)" value={lam} min={0} max={1} step={0.01}
              onChange={setLam} format={(v) => v.toFixed(2)} accent={ACCENT} />
            <Slider label="V(s) — constant critic guess" value={constantValue} min={0} max={2}
              step={0.05} onChange={setConstantValue} format={(v) => v.toFixed(2)} accent={ACCENT} />
            <div className="flex flex-wrap gap-1.5">
              <PresetButton accent={ACCENT} active={!terminated}
                onClick={() => setTerminated(false)}>
                truncated (time cap)
              </PresetButton>
              <PresetButton accent={ACCENT} active={terminated}
                onClick={() => setTerminated(true)}>
                terminated (done=1)
              </PresetButton>
            </div>
            {!terminated && (
              <Slider label="bootstrap V(s_next) after the cap" value={bootstrap}
                min={0} max={2} step={0.05} onChange={setBootstrap}
                format={(v) => v.toFixed(2)} accent={ACCENT} />
            )}
            <Eq>δₜ = rₜ + γ·V(sₜ₊₁)·(1−dₜ) − V(sₜ)</Eq>
            <Eq>Aₜ = δₜ + γλ·(1−dₜ)·Aₜ₊₁</Eq>
            <p className="font-mono text-[11px] leading-relaxed text-slate-400">
              λ=0 collapses to one-step TD error; λ=1 is a full discounted
              return minus the baseline. λ=0.95 lets credit from the reward at
              t=10 flow back across earlier steps without full Monte-Carlo
              variance.
            </p>
          </div>
          <div className="min-w-0 space-y-3">
            <div>
              <div className="mb-1 font-mono text-[10px] text-slate-500">
                rewards rₜ — click a bar slot to cycle 0 → +1 → +2 → −1
              </div>
              <div className="flex h-16 items-end gap-[2px] rounded-lg bg-[#070b18] px-1 pb-1 pt-1">
                {rewards.map((value, index) => (
                  <button key={index} onClick={() => cycleReward(index)}
                    className="relative h-full flex-1" title={`t=${index}: r=${value}`}>
                    <div
                      className="absolute bottom-0 left-[15%] right-[15%] rounded-[2px]"
                      style={{
                        background: value >= 0 ? '#4ade80' : '#f87171',
                        height: `${Math.abs(value) * 26 + (value !== 0 ? 4 : 2)}px`,
                        opacity: value === 0 ? 0.25 : 0.95,
                      }}
                    />
                  </button>
                ))}
              </div>
            </div>
            <Bars values={deltas} color="#94a3b8" label="TD errors δₜ" />
            <Bars values={advantages} color={ACCENT} label="advantages Aₜ (what PPO actually weighs)" />
            <div className="font-mono text-[10px] text-slate-500">
              tail: {terminated
                ? 'done=1 masks the bootstrap — value beyond the end is asserted to be 0'
                : `done=0, bootstrap γ·V(s_next)=${(gamma * bootstrap).toFixed(3)} keeps flowing through the cap`}
            </div>
          </div>
        </div>
        <RepoRefs
          refs={[
            ['agent_core/ppo_trainer.py', '2223-2247'],
            ['config.yaml', 'gamma: 0.99 · gae_lambda default 0.95'],
          ]}
          note={
            'The repo stores time caps as truncations, not terminals, so the ' +
            'bootstrap keeps flowing (flip the toggle and watch the late-step ' +
            'advantages change sign). GAE runs per fragment with a bootstrap tail — ' +
            'this strip IS one fragment.'
          }
        />
      </Card>
    </div>
  )
}
