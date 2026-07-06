// Math Lab: interactive playgrounds for the core equations the agent
// actually runs. Each page is grounded in this repo (file:line refs in
// its footer); synthetic stand-ins are labeled as such.
import { useState } from 'react'
import { LifPlayground } from './LifPlayground.jsx'
import { SurrogatePlayground } from './SurrogatePlayground.jsx'
import { GaePlayground } from './GaePlayground.jsx'
import { ClipPlayground } from './ClipPlayground.jsx'
import { TargetPlayground } from './TargetPlayground.jsx'

const PAGES = [
  {
    id: 'lif',
    nav: 'LIF neuron',
    title: 'Synaptic LIF dynamics',
    subtitle: 'the (syn, mem) state both token pathways carry across env steps',
    accent: '#c084fc',
    component: LifPlayground,
  },
  {
    id: 'surrogate',
    nav: 'surrogate grad',
    title: 'Surrogate gradient',
    subtitle: 'hard spike forward, fast-sigmoid gradient backward',
    accent: '#f472b6',
    component: SurrogatePlayground,
  },
  {
    id: 'gae',
    nav: 'GAE',
    title: 'Generalized Advantage Estimation',
    subtitle: 'per-fragment credit assignment, truncation ≠ termination',
    accent: '#fbbf24',
    component: GaePlayground,
  },
  {
    id: 'clip',
    nav: 'PPO clip',
    title: 'PPO clipped surrogate',
    subtitle: 'why updates stop pushing past the trust region',
    accent: '#f59e0b',
    component: ClipPlayground,
  },
  {
    id: 'target',
    nav: 'coarse→fine',
    title: 'Coarse-to-fine spatial target',
    subtitle: '49-way cell × 144-way offset instead of 7 056 pixels',
    accent: '#22d3ee',
    component: TargetPlayground,
  },
]

export function MathLab({ runData, page, onPageChange, onClose }) {
  const [fallbackPage, setFallbackPage] = useState(PAGES[0].id)
  const activeId = page ?? fallbackPage
  const setPage = onPageChange ?? setFallbackPage
  const active = PAGES.find((candidate) => candidate.id === activeId) ?? PAGES[0]
  const ActiveComponent = active.component

  return (
    <div className="absolute inset-0 z-40 flex flex-col bg-[#050510]/92 backdrop-blur-sm">
      <header className="flex items-start justify-between px-6 pb-2 pt-5">
        <div>
          <h1 className="font-mono text-sm font-bold uppercase tracking-[0.3em] text-cyan-300">
            Math Lab
          </h1>
          <p className="mt-1 font-mono text-[10px] text-slate-500">
            the agent&apos;s equations, runnable · every page cites its source in
            this repo
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

      <div className="flex min-h-0 flex-1 gap-4 px-6 pb-6 pt-2">
        <nav className="flex w-44 shrink-0 flex-col gap-1.5">
          {PAGES.map((candidate) => (
            <button
              key={candidate.id}
              onClick={() => setPage(candidate.id)}
              className={`rounded-xl border px-3 py-2 text-left font-mono text-[11px] transition
                          ${candidate.id === active.id ? 'text-slate-100' : 'text-slate-400 hover:text-slate-200'}`}
              style={{
                borderColor:
                  candidate.id === active.id
                    ? `${candidate.accent}66`
                    : 'rgba(100,116,139,0.25)',
                background:
                  candidate.id === active.id ? `${candidate.accent}0f` : 'transparent',
              }}
            >
              {candidate.nav}
            </button>
          ))}
        </nav>

        <main className="neon-scroll min-w-0 flex-1 overflow-y-auto pr-1">
          <div className="mb-3">
            <h2 className="text-lg font-semibold text-slate-100">{active.title}</h2>
            <p className="text-xs text-slate-400">{active.subtitle}</p>
          </div>
          <ActiveComponent runData={runData} />
        </main>
      </div>
    </div>
  )
}
