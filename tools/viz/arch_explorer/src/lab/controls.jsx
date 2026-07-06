// Shared UI atoms for the math-lab playgrounds.

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format = (v) => v,
  accent = '#22d3ee',
}) {
  return (
    <label className="block font-mono text-[11px] text-slate-400">
      <span className="flex justify-between">
        <span>{label}</span>
        <span className="text-slate-100">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full"
        style={{ accentColor: accent }}
      />
    </label>
  )
}

export function PresetButton({ children, onClick, active = false, accent }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-lg border px-2.5 py-1 font-mono text-[10px] transition
                  ${active ? 'text-slate-100' : 'text-slate-400 hover:text-slate-200'}`}
      style={{
        borderColor: active ? `${accent}88` : 'rgba(100,116,139,0.35)',
        background: active ? `${accent}14` : 'transparent',
      }}
    >
      {children}
    </button>
  )
}

export function Card({ title, accent, children }) {
  return (
    <div
      className="glass rounded-2xl p-4"
      style={{ borderColor: `${accent}33` }}
    >
      {title && (
        <h4
          className="mb-3 font-mono text-[10px] uppercase tracking-[0.2em]"
          style={{ color: accent }}
        >
          {title}
        </h4>
      )}
      {children}
    </div>
  )
}

// "Grounded in this repo" footer: file:line references + optional note.
export function RepoRefs({ refs, note }) {
  return (
    <div className="mt-3 border-t border-slate-700/40 pt-2 font-mono text-[10px] text-slate-500">
      {refs.map(([file, lines]) => (
        <div key={`${file}-${lines}`}>
          <span className="text-slate-400">{file}</span>
          {lines ? `:${lines}` : ''}
        </div>
      ))}
      {note && <div className="mt-1 leading-relaxed">{note}</div>}
    </div>
  )
}

// Formula block matching the info panel's .math-html .eq styling.
export function Eq({ children }) {
  return <div className="math-html"><p className="eq">{children}</p></div>
}
