// Zone stations: platform + signature geometry + neon edges + label.
// Hover lifts emissive intensity; click focuses the camera and opens the
// info panel. Labels/tooltips use drei <Html> (DOM), so no font assets.
import { useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import { Edges, Html } from '@react-three/drei'
import * as THREE from 'three'
import { getGlowTexture } from './glow'

function SignatureGeometry({ kind }) {
  switch (kind) {
    case 'globe':
      return <icosahedronGeometry args={[2.4, 1]} />
    case 'filter':
      return <coneGeometry args={[2.4, 3.6, 6]} />
    case 'crate':
      return <boxGeometry args={[3.2, 3.2, 3.2]} />
    case 'prisms':
      return <dodecahedronGeometry args={[2.4, 0]} />
    case 'ring':
      return <torusGeometry args={[2.4, 0.55, 12, 48]} />
    case 'spikes':
      return <octahedronGeometry args={[2.6, 0]} />
    case 'pathways':
      return <torusKnotGeometry args={[1.9, 0.5, 96, 12, 2, 3]} />
    case 'trident':
      return <cylinderGeometry args={[0.5, 2.2, 4.2, 3]} />
    case 'gate':
      return <torusGeometry args={[2.2, 0.35, 8, 4]} />
    default:
      return <sphereGeometry args={[2.2, 24, 24]} />
  }
}

export function Station({ zone, position, selected, onSelect }) {
  const coreRef = useRef()
  const materialRef = useRef()
  const [hovered, setHovered] = useState(false)
  const glowMap = useMemo(() => getGlowTexture(), [])
  const active = hovered || selected

  useFrame((state, delta) => {
    const core = coreRef.current
    const material = materialRef.current
    if (!core || !material) return
    core.rotation.y += delta * (active ? 0.9 : 0.25)
    const targetIntensity = active ? 2.2 : 0.75
    material.emissiveIntensity +=
      (targetIntensity - material.emissiveIntensity) * Math.min(1, delta * 8)
    const targetScale = active ? 1.14 : 1.0
    const s = core.scale.x + (targetScale - core.scale.x) * Math.min(1, delta * 8)
    core.scale.setScalar(s)
  })

  return (
    <group position={position}>
      {/* platform */}
      <mesh position={[0, -2.6, 0]}>
        <cylinderGeometry args={[4.2, 4.6, 0.4, 32]} />
        <meshStandardMaterial
          color="#0b1020"
          emissive={zone.color}
          emissiveIntensity={0.08}
          metalness={0.6}
          roughness={0.4}
        />
        <Edges color={zone.color} threshold={30} />
      </mesh>

      {/* halo sprite */}
      <sprite scale={[13, 13, 1]} position={[0, 0.4, 0]}>
        <spriteMaterial
          map={glowMap}
          color={zone.color}
          transparent
          opacity={active ? 0.55 : 0.3}
          depthWrite={false}
          blending={THREE.AdditiveBlending}
        />
      </sprite>

      {/* signature core */}
      <mesh
        ref={coreRef}
        position={[0, 0.4, 0]}
        onClick={(event) => {
          event.stopPropagation()
          onSelect(zone)
        }}
        onPointerOver={(event) => {
          event.stopPropagation()
          setHovered(true)
          document.body.style.cursor = 'pointer'
        }}
        onPointerOut={() => {
          setHovered(false)
          document.body.style.cursor = 'auto'
        }}
      >
        <SignatureGeometry kind={zone.geometry} />
        <meshStandardMaterial
          ref={materialRef}
          color="#0a0f1e"
          emissive={zone.color}
          emissiveIntensity={0.75}
          metalness={0.35}
          roughness={0.3}
          transparent
          opacity={0.92}
        />
        <Edges color={zone.color} threshold={15} />
      </mesh>

      <pointLight color={zone.color} intensity={active ? 60 : 22} distance={22} />

      {/* label + hover tooltip with tensor shapes */}
      <Html position={[0, 4.6, 0]} center distanceFactor={30} zIndexRange={[10, 0]}>
        <div className="station-label" style={{ color: zone.color }}>
          {zone.title}
        </div>
      </Html>
      {hovered && !selected && (
        <Html position={[0, 6.6, 0]} center distanceFactor={26} zIndexRange={[20, 0]}>
          <div className="station-tooltip">
            {zone.io.out.slice(0, 3).map(([name, shape]) => (
              <div key={name}>
                <span style={{ color: zone.color }}>{name}</span>
                <span className="dim"> {shape}</span>
              </div>
            ))}
            <div className="dim">click to inspect</div>
          </div>
        </Html>
      )}
    </group>
  )
}

export function CurveTube({ curve, color, radius = 0.14, opacity = 0.35 }) {
  const geometry = useMemo(
    () => new THREE.TubeGeometry(curve, 256, radius, 8, false),
    [curve, radius],
  )
  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial
        color={color}
        transparent
        opacity={opacity}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </mesh>
  )
}
