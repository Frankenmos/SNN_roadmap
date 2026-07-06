// Slide-out glassmorphic info panel: real shapes, real math, real code
// with file:line references. Content comes from src/data/zones.js and is
// authored from this repository only.
export function InfoPanel({ zone, zones, onSelect, onClose }) {
  if (!zone) return null
  const index = zones.findIndex((z) => z.id === zone.id)
  const prev = zones[(index - 1 + zones.length) % zones.length]
  const next = zones[(index + 1) % zones.length]

  return (
    <aside
      className="glass absolute right-4 top-4 bottom-4 z-30 flex w-[430px] max-w-[94vw]
                 flex-col rounded-2xl"
      style={{ borderColor: `${zone.color}44` }}
    >
      <header
        className="flex items-start justify-between gap-3 border-b px-5 py-4"
        style={{ borderColor: `${zone.color}22` }}
      >
        <div>
          <div
            className="font-mono text-[10px] uppercase tracking-[0.25em]"
            style={{ color: zone.color }}
          >
            zone {index + 1} / {zones.length}
          </div>
          <h2 className="mt-1 text-lg font-semibold text-slate-100">{zone.title}</h2>
          <p className="mt-0.5 text-xs text-slate-400">{zone.subtitle}</p>
        </div>
        <button
          onClick={onClose}
          className="rounded-lg border border-slate-600/40 px-2 py-1 font-mono text-xs
                     text-slate-400 transition hover:border-slate-400 hover:text-slate-100"
          title="Close (Esc)"
        >
          ✕
        </button>
      </header>

      <div className="neon-scroll flex-1 space-y-5 overflow-y-auto px-5 py-4">
        <section>
          <h3 className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">
            tensors in / out
          </h3>
          <table className="w-full text-left font-mono text-[11px]">
            <tbody>
              {zone.io.in.map(([name, shape]) => (
                <tr key={`in-${name}`} className="align-top">
                  <td className="w-8 pr-2 text-cyan-600">in</td>
                  <td className="pr-2 text-slate-200">{name}</td>
                  <td className="text-slate-400">{shape}</td>
                </tr>
              ))}
              {zone.io.out.map(([name, shape]) => (
                <tr key={`out-${name}`} className="align-top">
                  <td className="w-8 pr-2" style={{ color: zone.color }}>
                    out
                  </td>
                  <td className="pr-2 text-slate-200">{name}</td>
                  <td className="text-slate-400">{shape}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section>
          <h3 className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">
            the math
          </h3>
          {/* Local, hand-authored HTML from src/data/zones.js - not user input */}
          <div className="math-html" dangerouslySetInnerHTML={{ __html: zone.math }} />
        </section>

        <section>
          <h3 className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">
            why it is built this way
          </h3>
          <p className="text-[13px] leading-relaxed text-slate-300">{zone.why}</p>
        </section>

        <section>
          <h3 className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-slate-500">
            the actual source
          </h3>
          <div
            className="overflow-hidden rounded-lg border"
            style={{ borderColor: `${zone.color}33` }}
          >
            <div
              className="flex items-center justify-between border-b bg-slate-900/70 px-3 py-1.5
                         font-mono text-[10px]"
              style={{ borderColor: `${zone.color}22` }}
            >
              <span style={{ color: zone.color }}>{zone.code.file}</span>
              <span className="text-slate-500">lines {zone.code.lines}</span>
            </div>
            <pre
              className="neon-scroll overflow-x-auto bg-[#070b18] p-3 font-mono text-[11px]
                         leading-relaxed text-slate-300"
            >
              {zone.code.text}
            </pre>
          </div>
        </section>
      </div>

      <footer
        className="flex items-center justify-between border-t px-5 py-3"
        style={{ borderColor: `${zone.color}22` }}
      >
        <button
          onClick={() => onSelect(prev)}
          className="rounded-lg border border-slate-600/40 px-3 py-1.5 font-mono text-xs
                     text-slate-300 transition hover:border-slate-400 hover:text-white"
        >
          ← {prev.title}
        </button>
        <button
          onClick={() => onSelect(next)}
          className="rounded-lg border border-slate-600/40 px-3 py-1.5 font-mono text-xs
                     text-slate-300 transition hover:border-slate-400 hover:text-white"
        >
          {next.title} →
        </button>
      </footer>
    </aside>
  )
}
