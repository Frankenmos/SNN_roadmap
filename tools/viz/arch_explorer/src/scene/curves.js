// Spline layout: the inference pipeline snakes along X with a gentle
// S-curve in Z; the training loop is a closed circuit floating above it.
import * as THREE from 'three'
import { PIPELINE_ZONES, TRAINING_ZONES } from '../data/zones'

const PIPELINE_SPAN = 150

export const pipelineStations = PIPELINE_ZONES.map((zone, index) => {
  const n = PIPELINE_ZONES.length
  return new THREE.Vector3(
    -PIPELINE_SPAN / 2 + (index * PIPELINE_SPAN) / (n - 1),
    0,
    Math.sin(index * 1.05) * 10,
  )
})

export const pipelineCurve = new THREE.CatmullRomCurve3(
  pipelineStations,
  false,
  'catmullrom',
  0.5,
)

// Camera rail: the pipeline curve pushed up and back.
export const cameraCurve = new THREE.CatmullRomCurve3(
  pipelineStations.map((p) => p.clone().add(new THREE.Vector3(0, 11, 26))),
  false,
  'catmullrom',
  0.5,
)

// Arc-length parameter of the point on the curve nearest each station
// (CatmullRom control points do not sit at uniform t).
const SAMPLES = 600
export const stationTs = pipelineStations.map((station) => {
  let bestT = 0
  let bestD = Infinity
  for (let i = 0; i <= SAMPLES; i += 1) {
    const t = i / SAMPLES
    const d = pipelineCurve.getPointAt(t).distanceToSquared(station)
    if (d < bestD) {
      bestD = d
      bestT = t
    }
  }
  return bestT
})

// Training loop: closed ellipse above the middle of the pipeline.
const LOOP_CENTER = new THREE.Vector3(0, 24, -4)
export const trainingStations = TRAINING_ZONES.map((zone, index) => {
  const angle = (index / TRAINING_ZONES.length) * Math.PI * 2 + Math.PI / 2
  return new THREE.Vector3(
    LOOP_CENTER.x + Math.cos(angle) * 34,
    LOOP_CENTER.y + Math.sin(index * 2.1) * 1.5,
    LOOP_CENTER.z + Math.sin(angle) * 16,
  )
})

export const trainingCurve = new THREE.CatmullRomCurve3(
  trainingStations,
  true,
  'catmullrom',
  0.5,
)

// Precomputed position lookup tables for the particle flows.
export function buildCurveLUT(curve, resolution = 512) {
  const table = new Float32Array((resolution + 1) * 3)
  for (let i = 0; i <= resolution; i += 1) {
    const p = curve.getPointAt(i / resolution)
    table[i * 3] = p.x
    table[i * 3 + 1] = p.y
    table[i * 3 + 2] = p.z
  }
  return { table, resolution }
}

export function sampleLUT(lut, t, out) {
  const clamped = t - Math.floor(t)
  const f = clamped * lut.resolution
  const i = Math.floor(f)
  const frac = f - i
  const a = i * 3
  const b = Math.min(i + 1, lut.resolution) * 3
  out.set(
    lut.table[a] + (lut.table[b] - lut.table[a]) * frac,
    lut.table[a + 1] + (lut.table[b + 1] - lut.table[a + 1]) * frac,
    lut.table[a + 2] + (lut.table[b + 2] - lut.table[a + 2]) * frac,
  )
  return out
}
