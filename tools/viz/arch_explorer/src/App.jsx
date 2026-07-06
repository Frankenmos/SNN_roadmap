import { useCallback, useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import * as THREE from 'three'
import { PIPELINE_ZONES, TRAINING_ZONES } from './data/zones'
import { pipelineStations, trainingStations } from './scene/curves'
import { Scene } from './scene/Scene.jsx'
import { InfoPanel } from './ui/InfoPanel.jsx'
import { HUD } from './ui/HUD.jsx'

export default function App() {
  const [selected, setSelected] = useState(null)
  const [focus, setFocus] = useState(null)
  const [showTraining, setShowTraining] = useState(false)

  // Per-zone camera poses: hover point above/front of each station.
  const poses = useMemo(() => {
    const map = {}
    PIPELINE_ZONES.forEach((zone, index) => {
      const station = pipelineStations[index]
      map[zone.id] = {
        position: station.clone().add(new THREE.Vector3(-3, 6, 18)),
        lookAt: station.clone(),
      }
    })
    TRAINING_ZONES.forEach((zone, index) => {
      const station = trainingStations[index]
      map[zone.id] = {
        position: station.clone().add(new THREE.Vector3(0, 5, 17)),
        lookAt: station.clone(),
      }
    })
    return map
  }, [])

  const handleSelect = useCallback(
    (zone) => {
      setSelected(zone)
      setFocus(poses[zone.id])
      if (TRAINING_ZONES.some((z) => z.id === zone.id)) {
        setShowTraining(true)
      }
    },
    [poses],
  )

  const handleClose = useCallback(() => {
    setSelected(null)
    setFocus(null)
  }, [])

  // Scrolling releases the camera but keeps the panel open for reading.
  const handleScrollTakeover = useCallback(() => setFocus(null), [])

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === 'Escape') handleClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [handleClose])

  // Debug/smoke-test hook: select a zone by id without pointer raycasts.
  useEffect(() => {
    window.__ARCH_EXPLORER_SELECT = (zoneId) => {
      const zone = [...PIPELINE_ZONES, ...TRAINING_ZONES].find(
        (candidate) => candidate.id === zoneId,
      )
      if (zone) handleSelect(zone)
      return Boolean(zone)
    }
    return () => {
      delete window.__ARCH_EXPLORER_SELECT
    }
  }, [handleSelect])

  const panelZones = useMemo(() => {
    if (!selected) return PIPELINE_ZONES
    return TRAINING_ZONES.some((z) => z.id === selected.id)
      ? TRAINING_ZONES
      : PIPELINE_ZONES
  }, [selected])

  return (
    <div className="relative h-full w-full bg-[#050510] text-slate-200">
      <Canvas
        gl={{ antialias: true, preserveDrawingBuffer: true }}
        camera={{ fov: 50, near: 0.5, far: 600, position: [-95, 16, 45] }}
        onPointerMissed={handleClose}
      >
        <Scene
          selectedId={selected?.id ?? null}
          onSelect={handleSelect}
          showTraining={showTraining}
          focus={focus}
          onScrollTakeover={handleScrollTakeover}
        />
      </Canvas>

      <HUD
        showTraining={showTraining}
        onToggleTraining={() => setShowTraining((value) => !value)}
      />
      <InfoPanel
        zone={selected}
        zones={panelZones}
        onSelect={handleSelect}
        onClose={handleClose}
      />
    </div>
  )
}
