import { TOKEN_GROUPS } from '../data/zones'

export function HUD({ showTraining, onToggleTraining, runData }) {
  return (
    <>
      {/* title */}
      <div className="glass absolute left-4 top-4 z-20 rounded-2xl px-5 py-3">
        <h1 className="font-mono text-sm font-bold uppercase tracking-[0.3em] text-cyan-300">
          SNN-PPO Architecture Explorer
        </h1>
        <p className="mt-1 font-mono text-[10px] text-slate-400">
          DefeatRoaches agent · 95-token spiking policy · content verified
          against this repo (2026-07-06)
        </p>
        <p className="mt-2 font-mono text-[10px] text-slate-500">
          scroll = dolly · hover = shapes · click = inspect · esc = close
        </p>
      </div>

      {/* training loop toggle */}
      <button
        onClick={onToggleTraining}
        className={`glass absolute right-4 top-4 z-20 rounded-2xl px-4 py-3 font-mono text-xs
                    uppercase tracking-widest transition
                    ${showTraining ? 'text-amber-300' : 'text-slate-400 hover:text-slate-200'}`}
        style={showTraining ? { borderColor: 'rgba(251,191,36,0.4)' } : undefined}
      >
        {showTraining ? '◉' : '○'} training loop
      </button>

      {/* live run-data badge (written by `python -m tools.registry export`) */}
      {runData && (
        <div className="glass absolute bottom-4 right-4 z-20 rounded-2xl px-4 py-3">
          <div className="mb-1 flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-400/80">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
            live run data
          </div>
          <div className="max-w-[240px] truncate font-mono text-[11px] text-slate-200">
            {runData.run}
          </div>
          <div className="mt-0.5 font-mono text-[10px] text-slate-500">
            {runData.entries.length} artifact
            {runData.entries.length === 1 ? '' : 's'} · exported{' '}
            {(runData.generated_iso ?? '').slice(0, 10) || 'n/a'}
          </div>
        </div>
      )}

      {/* token legend */}
      <div className="glass absolute bottom-4 left-4 z-20 rounded-2xl px-4 py-3">
        <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">
          95-token stream
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {TOKEN_GROUPS.map((group) => (
            <div key={group.name} className="flex items-center gap-1.5 font-mono text-[10px]">
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ background: group.color }}
              />
              <span className="text-slate-300">
                {group.count} {group.name}
              </span>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
