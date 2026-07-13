// "Live values" section of the info panel: real numbers from an exported
// run bundle (see src/data/runData.js). Zone renderers are plain
// functions (no hooks) so the section can hide itself entirely when a
// zone has no live slice - no empty scaffolding.
import {
  entryLabel,
  formatCount,
  formatFloat,
  pathwayTimeConstants,
} from '../data/runData'

function Sparkline({ values, color = '#22d3ee', height = 26 }) {
  const width = 150
  const valid = []
  values.forEach((value, index) => {
    if (value != null && Number.isFinite(value)) valid.push([index, value])
  })
  if (valid.length < 2) return null
  let min = Infinity
  let max = -Infinity
  for (const [, value] of valid) {
    if (value < min) min = value
    if (value > max) max = value
  }
  const span = max - min || 1
  const denom = Math.max(values.length - 1, 1)
  const points = valid
    .map(
      ([index, value]) =>
        `${((index / denom) * width).toFixed(1)},${(
          height - 2 - ((value - min) / span) * (height - 4)
        ).toFixed(1)}`,
    )
    .join(' ')
  return (
    <svg width={width} height={height} className="block" aria-hidden="true">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.2"
        opacity="0.9"
      />
    </svg>
  )
}

function snnLive({ runData, entry }) {
  const constants = pathwayTimeConstants(entry)
  const init = runData.config?.snn_init ?? {}
  const rows = []
  for (const pathway of ['fast', 'slow']) {
    for (const kind of ['alpha', 'beta']) {
      const row = constants[pathway][kind]
      if (row) rows.push({ pathway, kind, init: init[`${pathway}_${kind}`], row })
    }
  }
  if (!rows.length) return null
  const slowBeta = constants.slow.beta
  const integrator = slowBeta && slowBeta.effective_mean >= 0.999
  return (
    <div className="space-y-2">
      <table className="w-full text-left font-mono text-[11px]">
        <thead>
          <tr className="text-slate-500">
            <th className="font-normal">pathway</th>
            <th className="font-normal">const</th>
            <th className="text-right font-normal">init</th>
            <th className="text-right font-normal">learned</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ pathway, kind, init: initValue, row }) => (
            <tr key={`${pathway}-${kind}`} className="align-top">
              <td className={pathway === 'fast' ? 'text-pink-300' : 'text-violet-300'}>
                {pathway}
              </td>
              <td className="text-slate-300">{kind === 'alpha' ? 'α syn' : 'β mem'}</td>
              <td className="text-right text-slate-400">{formatFloat(initValue, 2)}</td>
              <td className="text-right text-slate-100">
                {formatFloat(row.effective_mean)}
                {Number.isFinite(row.mean) &&
                  Math.abs(row.mean - row.effective_mean) > 5e-4 && (
                    <span className="text-slate-500"> raw {formatFloat(row.mean)}</span>
                  )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {integrator && (
        <p className="text-[11px] leading-relaxed text-violet-200/80">
          The slow pathway has pushed β to the clamp — effectively a no-leak
          integrator (snnTorch clamps α/β to [0, 1] in the forward pass).
          The outer ring in the scene barely decays for the same reason.
        </p>
      )}
    </div>
  )
}

function dispatchLive({ runData, entry }) {
  const row = entry.update_row
  if (!row) return null
  const mix = [
    ['no_op', row.rollout_policy_no_op_count, '#64748b'],
    ['left_click', row.rollout_policy_left_click_count, '#f59e0b'],
    ['right_click', row.rollout_policy_right_click_count, '#22d3ee'],
  ].filter(([, value]) => value != null)
  const total = mix.reduce((sum, [, value]) => sum + value, 0)
  if (!total) return null

  const series = runData.history?.series
  let shareHistory = null
  if (
    series?.rollout_policy_right_click_count &&
    series?.rollout_policy_no_op_count
  ) {
    shareHistory = series.rollout_policy_right_click_count.map((right, i) => {
      const noop = series.rollout_policy_no_op_count[i]
      const left = series.rollout_policy_left_click_count?.[i] ?? 0
      const sum = (right ?? 0) + (noop ?? 0) + (left ?? 0)
      return right != null && sum > 0 ? right / sum : null
    })
  }

  return (
    <div className="space-y-2 font-mono text-[11px]">
      <div className="flex h-2.5 overflow-hidden rounded-full bg-slate-800/60">
        {mix.map(([name, value, color]) => (
          <div
            key={name}
            style={{ width: `${(100 * value) / total}%`, background: color }}
          />
        ))}
      </div>
      <table className="w-full text-left">
        <tbody>
          {mix.map(([name, value, color]) => (
            <tr key={name}>
              <td style={{ color }}>{name}</td>
              <td className="text-right text-slate-200">{formatCount(value)}</td>
              <td className="w-14 text-right text-slate-500">
                {((100 * value) / total).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {(row.rollout_feedback_near_enemy_smart_count != null ||
        row.rollout_feedback_enemy_health_drop_after_smart_count != null) && (
        <div className="text-slate-400">
          confirmed engagements: near-enemy{' '}
          {formatCount(row.rollout_feedback_near_enemy_smart_count)} · health-drop{' '}
          {formatCount(row.rollout_feedback_enemy_health_drop_after_smart_count)}
        </div>
      )}
      {shareHistory && (
        <div>
          <div className="text-slate-500">right-click share across training</div>
          <Sparkline values={shareHistory} color="#22d3ee" />
        </div>
      )}
      <div className="text-slate-500">
        rollout mix at update {formatCount(row.global_update_index)}
      </div>
    </div>
  )
}

function headsLive({ runData }) {
  const counts = runData.module_param_counts
  if (!counts || !Object.keys(counts).length) return null
  const rows = Object.entries(counts)
  const total = rows.reduce((sum, [, value]) => sum + value, 0)
  const top = rows.slice(0, 8)
  return (
    <div className="space-y-1 font-mono text-[11px]">
      {top.map(([module, count]) => (
        <div key={module} className="flex items-center gap-2">
          <span className="w-40 truncate text-slate-300">{module}</span>
          <span className="h-1.5 rounded-full bg-cyan-400/70"
            style={{ width: `${Math.max(2, (72 * count) / top[0][1])}px` }}
          />
          <span className="ml-auto text-slate-400">{formatCount(count)}</span>
        </div>
      ))}
      <div className="pt-1 text-slate-500">
        total {formatCount(total)} params · {rows.length} modules
      </div>
    </div>
  )
}

function ppoLive({ runData, entry }) {
  const row = entry.update_row
  if (!row) return null
  const scalars = [
    ['entropy', row.mean_entropy, 4],
    ['KL', row.mean_kl, 4],
    ['clip fraction', row.clip_fraction, 3],
    ['grad norm', row.grad_norm, 3],
  ].filter(([, value]) => value != null)
  const decomposition = [
    ['trunk', row.grad_norm_trunk],
    ['actor head', row.grad_norm_actor_head],
    ['critic head', row.grad_norm_critic_head],
    ['target head', row.grad_norm_target_head],
  ].filter(([, value]) => value != null)
  if (!scalars.length && !decomposition.length) return null
  const gradHistory = runData.history?.series?.grad_norm
  return (
    <div className="space-y-2 font-mono text-[11px]">
      <table className="w-full text-left">
        <tbody>
          {scalars.map(([label, value, digits]) => (
            <tr key={label}>
              <td className="text-slate-500">{label}</td>
              <td className="text-right text-slate-200">
                {formatFloat(value, digits)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {decomposition.length > 0 && (
        <div>
          <div className="mb-1 text-slate-500">grad-norm decomposition</div>
          <table className="w-full text-left">
            <tbody>
              {decomposition.map(([label, value]) => (
                <tr key={label}>
                  <td className="text-slate-300">{label}</td>
                  <td className="text-right text-slate-200">
                    {formatFloat(value, 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {gradHistory && (
        <div>
          <div className="text-slate-500">grad norm across training</div>
          <Sparkline values={gradHistory} color="#f59e0b" />
        </div>
      )}
      <div className="text-slate-500">
        at update {formatCount(row.global_update_index)}
      </div>
    </div>
  )
}

function silLive({ runData, entry }) {
  const row = entry.update_row
  if (!row) return null
  const scalars = [
    ['SIL loss', row.sil_loss, 4],
    ['gate open fraction', row.sil_gate_open_fraction, 3],
    ['buffer size', row.sil_buffer_size, 0],
    ['steps replayed', row.sil_steps_replayed, 0],
  ].filter(([, value]) => value != null)
  if (!scalars.length) return null
  const gateHistory = runData.history?.series?.sil_gate_open_fraction
  return (
    <div className="space-y-2 font-mono text-[11px]">
      <table className="w-full text-left">
        <tbody>
          {scalars.map(([label, value, digits]) => (
            <tr key={label}>
              <td className="text-slate-500">{label}</td>
              <td className="text-right text-slate-200">
                {digits === 0 ? formatCount(value) : formatFloat(value, digits)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {gateHistory && (
        <div>
          <div className="text-slate-500">gate open fraction across training</div>
          <Sparkline values={gateHistory} color="#4ade80" />
        </div>
      )}
      <div className="text-slate-500">
        at update {formatCount(row.global_update_index)}
      </div>
    </div>
  )
}

// Plain functions, called directly (not as JSX) so a null return can
// hide the whole section. None of them may use hooks.
const ZONE_RENDERERS = {
  snn: snnLive,
  dispatch: dispatchLive,
  heads: headsLive,
  ppo: ppoLive,
  sil: silLive,
}

export function LiveSection({ zone, runData, entryIndex, onEntryChange }) {
  if (!runData) return null
  const renderer = ZONE_RENDERERS[zone.id]
  if (!renderer) return null
  const entry =
    runData.entries[Math.min(entryIndex, runData.entries.length - 1)]
  const body = renderer({ runData, entry })
  if (!body) return null

  return (
    <section>
      <h3 className="mb-2 font-mono text-[10px] uppercase tracking-[0.2em] text-emerald-400/80">
        live · {runData.run}
      </h3>
      <select
        value={Math.min(entryIndex, runData.entries.length - 1)}
        onChange={(event) => onEntryChange(Number(event.target.value))}
        className="mb-2 w-full rounded-lg border border-slate-600/40 bg-slate-900/80
                   px-2 py-1.5 font-mono text-[11px] text-slate-200"
        title="Pick a snapshot/checkpoint - live values follow it"
      >
        {runData.entries.map((option, index) => (
          <option key={option.ref} value={index}>
            {entryLabel(option)}
          </option>
        ))}
      </select>
      <div className="mb-3 font-mono text-[10px] text-slate-500">
        {entry.episode != null && <>episode {formatCount(entry.episode)} · </>}
        {entry.eval_mean != null && (
          <>
            eval {formatFloat(entry.eval_mean, 1)}
            {entry.eval_policy_version != null &&
            entry.eval_policy_version !== entry.policy_version
              ? ` @u${entry.eval_policy_version}`
              : ''}
            {' · '}
          </>
        )}
        {entry.size_mib != null && <>{entry.size_mib} MiB</>}
      </div>
      {body}
    </section>
  )
}
