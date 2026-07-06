// Camera behavior: wheel scrubs a damped parameter along the camera
// rail (dolly through the pipeline input -> output); clicking a zone
// hands the camera a focus pose; scrolling takes it back. Mouse adds a
// subtle parallax. All vectors are reused - no per-frame allocation.
import { useEffect, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'
import { cameraCurve, pipelineCurve } from './curves'

const desiredPos = new THREE.Vector3()
const desiredLook = new THREE.Vector3()
const railPos = new THREE.Vector3()
const railLook = new THREE.Vector3()

export function CameraRig({ focus, onScrollTakeover }) {
  const scrollRef = useRef({ target: 0.02, current: 0.02 })
  const mouseRef = useRef({ x: 0, y: 0 })
  const currentPos = useRef(new THREE.Vector3(-90, 14, 40))
  const currentLook = useRef(new THREE.Vector3(-60, 0, 0))
  const { gl } = useThree()

  useEffect(() => {
    const onWheel = (event) => {
      scrollRef.current.target = THREE.MathUtils.clamp(
        scrollRef.current.target + event.deltaY * 0.00038,
        0,
        1,
      )
      onScrollTakeover()
    }
    const onPointerMove = (event) => {
      mouseRef.current.x = (event.clientX / window.innerWidth) * 2 - 1
      mouseRef.current.y = (event.clientY / window.innerHeight) * 2 - 1
    }
    const element = gl.domElement
    element.addEventListener('wheel', onWheel, { passive: true })
    window.addEventListener('pointermove', onPointerMove)
    return () => {
      element.removeEventListener('wheel', onWheel)
      window.removeEventListener('pointermove', onPointerMove)
    }
  }, [gl, onScrollTakeover])

  useFrame((state, delta) => {
    const scroll = scrollRef.current
    scroll.current += (scroll.target - scroll.current) * Math.min(1, delta * 3.2)

    if (focus) {
      desiredPos.copy(focus.position)
      desiredLook.copy(focus.lookAt)
    } else {
      cameraCurve.getPointAt(scroll.current, railPos)
      pipelineCurve.getPointAt(Math.min(1, scroll.current + 0.06), railLook)
      desiredPos.copy(railPos)
      desiredLook.copy(railLook)
    }
    desiredPos.x += mouseRef.current.x * 2.2
    desiredPos.y += -mouseRef.current.y * 1.4

    const damp = 1 - Math.exp(-delta * 3.0)
    currentPos.current.lerp(desiredPos, damp)
    currentLook.current.lerp(desiredLook, damp)
    state.camera.position.copy(currentPos.current)
    state.camera.lookAt(currentLook.current)
  })

  return null
}
