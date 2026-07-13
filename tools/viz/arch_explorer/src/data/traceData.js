// Optional trace bundle, written by:
//   python -m tools.analysis.trace_export <run> [--episode N] [--mode det]
// into public/trace_data.json. Same contract as runData: null on any
// miss, every consumer treats null as "no trace loaded".

export async function loadTraceData() {
  try {
    const response = await fetch(`${import.meta.env.BASE_URL}trace_data.json`, {
      cache: 'no-store',
    })
    if (!response.ok) return null
    const data = await response.json()
    if (
      !data ||
      data.kind !== 'arch-explorer-trace' ||
      data.schema_version !== 1 ||
      !Array.isArray(data.steps) ||
      data.steps.length === 0
    ) {
      return null
    }
    return data
  } catch {
    return null
  }
}

// base64(np.packbits(mask)) -> Uint8Array of 0/1, length size*size.
// numpy packbits is big-endian within each byte; row-major bit index
// = y * size + x (see tools/analysis/trace_export.py).
export function unpackMask(b64, size = 84) {
  const out = new Uint8Array(size * size)
  if (!b64) return out
  const raw = atob(b64)
  for (let i = 0; i < out.length; i += 1) {
    const byte = raw.charCodeAt(i >> 3)
    out[i] = (byte >> (7 - (i & 7))) & 1
  }
  return out
}

export const ACTION_LABELS = {
  0: 'no_op',
  1: 'left_click',
  2: 'right_click',
}
