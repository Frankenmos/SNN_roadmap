import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Grid, Stars } from '@react-three/drei'
import { PIPELINE_ZONES, TRAINING_ZONES } from '../data/zones'
import {
  pipelineCurve,
  pipelineStations,
  trainingCurve,
  trainingStations,
} from './curves'
import { Station, CurveTube } from './Stations.jsx'
import {
  FlowParticles,
  PathwayPulses,
  SpikeFlashes,
  TokenRing,
} from './Effects.jsx'
import { CameraRig } from './CameraRig.jsx'

// Signals the smoke test that real frames are being produced.
function ReadyFlag() {
  const framesRef = useRef(0)
  useFrame(() => {
    framesRef.current += 1
    if (framesRef.current === 8) {
      window.__ARCH_EXPLORER_READY = true
    }
  })
  return null
}

const streamStation = pipelineStations[PIPELINE_ZONES.findIndex((z) => z.id === 'stream')]
const attentionStation =
  pipelineStations[PIPELINE_ZONES.findIndex((z) => z.id === 'attention')]
const snnStation = pipelineStations[PIPELINE_ZONES.findIndex((z) => z.id === 'snn')]

export function Scene({
  selectedId,
  onSelect,
  showTraining,
  focus,
  onScrollTakeover,
  liveBetas = null,
}) {
  return (
    <>
      <color attach="background" args={['#050510']} />
      <fog attach="fog" args={['#050510', 60, 240]} />
      <ambientLight intensity={0.25} />
      <directionalLight position={[40, 60, 30]} intensity={0.5} color="#93c5fd" />
      <Stars radius={260} depth={80} count={2600} factor={5} saturation={0} fade speed={0.5} />
      <Grid
        position={[0, -3.4, 0]}
        args={[300, 300]}
        cellSize={4}
        cellThickness={0.4}
        cellColor="#101a33"
        sectionSize={20}
        sectionThickness={0.9}
        sectionColor="#155e75"
        fadeDistance={220}
        fadeStrength={1.5}
        infiniteGrid
      />

      {/* inference pipeline */}
      <CurveTube curve={pipelineCurve} color="#0ea5e9" />
      <FlowParticles curve={pipelineCurve} count={260} color="#7dd3fc" />
      {PIPELINE_ZONES.map((zone, index) => (
        <Station
          key={zone.id}
          zone={zone}
          position={pipelineStations[index]}
          selected={selectedId === zone.id}
          onSelect={onSelect}
        />
      ))}
      <TokenRing position={streamStation} />
      <SpikeFlashes position={attentionStation} />
      <PathwayPulses
        position={snnStation}
        fastBeta={liveBetas?.fast ?? null}
        slowBeta={liveBetas?.slow ?? null}
      />

      {/* training loop overlay */}
      {showTraining && (
        <group>
          <CurveTube curve={trainingCurve} color="#f59e0b" radius={0.12} opacity={0.3} />
          <FlowParticles curve={trainingCurve} count={110} color="#fcd34d" speed={0.04} />
          {TRAINING_ZONES.map((zone, index) => (
            <Station
              key={zone.id}
              zone={zone}
              position={trainingStations[index]}
              selected={selectedId === zone.id}
              onSelect={onSelect}
            />
          ))}
        </group>
      )}

      <CameraRig focus={focus} onScrollTakeover={onScrollTakeover} />
      <ReadyFlag />
    </>
  )
}
