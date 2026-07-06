// Optional live-run bundle, written by:
//   python -m tools.registry export <run>
// into public/run_data.json (dev) or dist/run_data.json (preview).
// The explorer must stay fully functional without it: the loader returns
// null on any miss and every consumer treats null as "static mode".

export async function loadRunData() {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}run_data.json`, {
      cache: 'no-store',
    })
    if (!response.ok) return null
    const data = await response.json()
    if (
      !data ||
      data.kind !== 'arch-explorer-run-data' ||
      data.schema_version !== 1 ||
      !Array.isArray(data.entries) ||
      data.entries.length === 0
    ) {
      return null
    }
    return data
  } catch {
    return null
  }
}

// Initial artifact to show: the final checkpoint, else the last entry.
export function defaultEntryIndex(data) {
  const index = data.entries.findIndex((entry) => entry.kind === 'checkpoint')
  return index >= 0 ? index : data.entries.length - 1
}

export function entryLabel(entry) {
  const version =
    entry.policy_version != null ? `u${entry.policy_version}` : 'u?'
  return entry.kind === 'snapshot'
    ? `${version} · snapshot`
    : `${version} · ${entry.kind}`
}

// State-dict names -> the fast/slow pathway slots the scene and panel
// talk about (agent_core/spiking_policy.py: token_snn = fast pathway,
// slow_token_snn = slow pathway; both are snn.Synaptic with learned
// alpha/beta). Unrecognized learnable constants land in `other`.
const PATHWAY_SLOTS = {
  'token_snn.snn.alpha': ['fast', 'alpha'],
  'token_snn.snn.beta': ['fast', 'beta'],
  'slow_token_snn.snn.alpha': ['slow', 'alpha'],
  'slow_token_snn.snn.beta': ['slow', 'beta'],
}

export function pathwayTimeConstants(entry) {
  const result = { fast: {}, slow: {}, other: [] }
  for (const row of entry?.time_constants ?? []) {
    const slot = PATHWAY_SLOTS[row.name]
    if (slot) {
      result[slot[0]][slot[1]] = row
    } else if (!row.name.startsWith('attention.')) {
      // attention lif betas are fixed at 0.5 (not learned) - skip them.
      result.other.push(row)
    }
  }
  return result
}

export function formatFloat(value, digits = 4) {
  if (value == null || !Number.isFinite(Number(value))) return 'n/a'
  return Number(value).toFixed(digits)
}

export function formatCount(value) {
  if (value == null || !Number.isFinite(Number(value))) return 'n/a'
  return Number(value).toLocaleString('en-US')
}
