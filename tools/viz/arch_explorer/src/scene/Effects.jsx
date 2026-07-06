// Animated matter: tensor-flow particles along the splines, the 95-token
// ring, the fast/slow SNN pathway pulses, and discrete attention spikes.
// Everything is instanced; per-frame work reuses module-scope temps and
// never allocates.
import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { buildCurveLUT, sampleLUT } from './curves'
import { TOKEN_GROUPS, SNN_TIME_CONSTANTS } from '../data/zones'

const tmpObject = new THREE.Object3D()
const tmpVec = new THREE.Vector3()

// Deterministic PRNG so headless renders are reproducible.
function mulberry32(seed) {
  let a = seed >>> 0
  return function next() {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function FlowParticles({ curve, count = 240, color = '#7dd3fc', speed = 0.028 }) {
  const meshRef = useRef()
  const lut = useMemo(() => buildCurveLUT(curve), [curve])
  const particles = useMemo(() => {
    const rand = mulberry32(1337)
    return Array.from({ length: count }, () => ({
      baseT: rand(),
      speed: speed * (0.6 + rand() * 0.9),
      lane: new THREE.Vector3(
        (rand() - 0.5) * 2.4,
        (rand() - 0.5) * 2.0 + 0.4,
        (rand() - 0.5) * 2.4,
      ),
      size: 0.12 + rand() * 0.2,
    }))
  }, [count, speed])

  useFrame((state) => {
    const mesh = meshRef.current
    if (!mesh) return
    const time = state.clock.elapsedTime
    for (let i = 0; i < particles.length; i += 1) {
      const p = particles[i]
      sampleLUT(lut, p.baseT + time * p.speed, tmpVec)
      tmpObject.position.copy(tmpVec).add(p.lane)
      tmpObject.scale.setScalar(p.size)
      tmpObject.rotation.set(0, 0, 0)
      tmpObject.updateMatrix()
      mesh.setMatrixAt(i, tmpObject.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
  })

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]} frustumCulled={false}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.85}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </instancedMesh>
  )
}

// The 95-token stream: one instanced cube per token, color-coded by
// token type (49 spatial / 24 entity / 20 selection / 1 feedback / 1
// meta), slowly orbiting its station as a double helix ribbon.
export function TokenRing({ position }) {
  const meshRef = useRef()
  const groupRef = useRef()
  const total = TOKEN_GROUPS.reduce((sum, group) => sum + group.count, 0)

  const tokens = useMemo(() => {
    const list = []
    let index = 0
    for (const group of TOKEN_GROUPS) {
      for (let k = 0; k < group.count; k += 1) {
        const t = index / total
        list.push({
          angle: t * Math.PI * 2,
          y: Math.sin(t * Math.PI * 4) * 1.6,
          radius: 5.4 + Math.cos(t * Math.PI * 6) * 0.5,
          color: new THREE.Color(group.color),
        })
        index += 1
      }
    }
    return list
  }, [total])

  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh) return
    for (let i = 0; i < tokens.length; i += 1) {
      mesh.setColorAt(i, tokens[i].color)
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }, [tokens])

  useFrame((state, delta) => {
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.22
    const mesh = meshRef.current
    if (!mesh) return
    const time = state.clock.elapsedTime
    for (let i = 0; i < tokens.length; i += 1) {
      const token = tokens[i]
      tmpObject.position.set(
        Math.cos(token.angle) * token.radius,
        token.y + Math.sin(time * 1.4 + token.angle * 3) * 0.25 + 0.4,
        Math.sin(token.angle) * token.radius,
      )
      tmpObject.rotation.set(0, token.angle, 0)
      tmpObject.scale.setScalar(0.34)
      tmpObject.updateMatrix()
      mesh.setMatrixAt(i, tmpObject.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
  })

  return (
    <group ref={groupRef} position={position}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, total]} frustumCulled={false}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          emissive="#ffffff"
          emissiveIntensity={0.55}
          color="#0a0f1e"
          toneMapped={false}
        />
      </instancedMesh>
    </group>
  )
}

// Fast vs slow SNN pathways: two counter-rotating orbit rings whose
// pulse rates and decay envelopes act out the alpha/beta story - the
// fast ring flashes at env-step tempo and forgets quickly, the slow
// ring pulses rarely and holds its charge.
export function PathwayPulses({ position }) {
  const fastMaterialRef = useRef()
  const slowMaterialRef = useRef()
  const fastOrbitRef = useRef()
  const slowOrbitRef = useRef()

  useFrame((state) => {
    const time = state.clock.elapsedTime
    // Pulse envelope ~ exp(-phase / tau): tau derived from beta (mem
    // decay): fast beta 0.65 -> rapid falloff; slow beta 0.97 -> long hold.
    const fastPhase = (time * 2.2) % 1
    const slowPhase = (time * 0.4) % 1
    if (fastMaterialRef.current) {
      fastMaterialRef.current.emissiveIntensity = 0.4 + 2.6 * Math.exp(-fastPhase * 6)
    }
    if (slowMaterialRef.current) {
      slowMaterialRef.current.emissiveIntensity = 0.5 + 2.0 * Math.exp(-slowPhase * 1.4)
    }
    if (fastOrbitRef.current) fastOrbitRef.current.rotation.y = time * 1.6
    if (slowOrbitRef.current) slowOrbitRef.current.rotation.y = -time * 0.35
  })

  const { fast, slow } = SNN_TIME_CONSTANTS
  return (
    <group position={position}>
      <group ref={fastOrbitRef}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[4.4, 0.1, 8, 64]} />
          <meshStandardMaterial
            ref={fastMaterialRef}
            color="#0a0f1e"
            emissive="#f472b6"
            emissiveIntensity={1}
            toneMapped={false}
          />
        </mesh>
        <mesh position={[4.4, 0, 0]}>
          <sphereGeometry args={[0.42, 10, 10]} />
          <meshBasicMaterial color="#f9a8d4" toneMapped={false} />
        </mesh>
      </group>
      <group ref={slowOrbitRef}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[6.4, 0.1, 8, 64]} />
          <meshStandardMaterial
            ref={slowMaterialRef}
            color="#0a0f1e"
            emissive="#a78bfa"
            emissiveIntensity={1}
            toneMapped={false}
          />
        </mesh>
        <mesh position={[6.4, 0, 0]}>
          <sphereGeometry args={[0.42, 10, 10]} />
          <meshBasicMaterial color="#ddd6fe" toneMapped={false} />
        </mesh>
      </group>
      {/* alpha/beta captions ride in the info panel; rings just encode rate */}
      {void fast}
      {void slow}
    </group>
  )
}

// Spiking attention: discrete spike FLASHES (sharp attack, fast decay,
// relocation every cycle) rather than continuous glow.
export function SpikeFlashes({ position, count = 42 }) {
  const meshRef = useRef()

  const { flashes, table } = useMemo(() => {
    const rand = mulberry32(4242)
    const tableSize = 512
    const positions = new Float32Array(tableSize * 3)
    for (let i = 0; i < tableSize; i += 1) {
      const radius = 3.2 + rand() * 2.6
      const theta = rand() * Math.PI * 2
      const phi = Math.acos(2 * rand() - 1)
      positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta)
      positions[i * 3 + 1] = radius * Math.cos(phi) * 0.7 + 0.4
      positions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta)
    }
    return {
      table: positions,
      flashes: Array.from({ length: count }, (_, i) => ({
        period: 0.7 + rand() * 1.3,
        phase: rand(),
        stride: 7 + Math.floor(rand() * 23),
        offset: i,
      })),
    }
  }, [count])

  useFrame((state) => {
    const mesh = meshRef.current
    if (!mesh) return
    const time = state.clock.elapsedTime
    for (let i = 0; i < flashes.length; i += 1) {
      const flash = flashes[i]
      const cycles = time / flash.period + flash.phase
      const frac = cycles - Math.floor(cycles)
      // Sharp attack (5% of cycle) then exponential decay: a spike, not a glow.
      const intensity = frac < 0.05 ? frac / 0.05 : Math.exp(-(frac - 0.05) * 9)
      const slot =
        ((flash.offset + Math.floor(cycles) * flash.stride) % 512 + 512) % 512
      tmpObject.position.set(
        table[slot * 3],
        table[slot * 3 + 1],
        table[slot * 3 + 2],
      )
      tmpObject.scale.setScalar(Math.max(0.0001, intensity * 0.55))
      tmpObject.rotation.set(0, 0, 0)
      tmpObject.updateMatrix()
      mesh.setMatrixAt(i, tmpObject.matrix)
    }
    mesh.instanceMatrix.needsUpdate = true
  })

  return (
    <group position={position}>
      <instancedMesh ref={meshRef} args={[undefined, undefined, count]} frustumCulled={false}>
        <octahedronGeometry args={[1, 0]} />
        <meshBasicMaterial
          color="#fda4af"
          transparent
          opacity={0.95}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </instancedMesh>
    </group>
  )
}
