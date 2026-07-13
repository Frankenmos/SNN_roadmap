// Procedural radial-gradient sprite texture (no external assets: the
// explorer must run fully offline).
import * as THREE from 'three'

let cached = null

export function getGlowTexture() {
  if (cached) return cached
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const gradient = ctx.createRadialGradient(
    size / 2, size / 2, 0,
    size / 2, size / 2, size / 2,
  )
  gradient.addColorStop(0, 'rgba(255,255,255,0.9)')
  gradient.addColorStop(0.25, 'rgba(255,255,255,0.35)')
  gradient.addColorStop(0.6, 'rgba(255,255,255,0.08)')
  gradient.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, size, size)
  cached = new THREE.CanvasTexture(canvas)
  return cached
}
