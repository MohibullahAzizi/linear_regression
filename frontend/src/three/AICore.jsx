import { useRef, useMemo, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Line, Sphere, Torus, Float } from '@react-three/drei'
import * as THREE from 'three'

function CoreSphere({ state = 'idle' }) {
  const ref = useRef()
  const colors = {
    idle: '#00E5CC',
    running: '#3987FF',
    completed: '#39D98A',
    warning: '#FFB84D',
    error: '#FF5263',
  }
  const color = colors[state] || colors.idle

  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.3
      ref.current.rotation.x += delta * 0.1
    }
  })

  return (
    <group ref={ref}>
      {/* Inner core */}
      <Sphere args={[0.6, 32, 32]}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.5}
          transparent
          opacity={0.85}
          roughness={0.2}
          metalness={0.8}
        />
      </Sphere>
      {/* Wireframe outer */}
      <Sphere args={[0.85, 16, 16]}>
        <meshBasicMaterial color={color} wireframe transparent opacity={0.15} />
      </Sphere>
    </group>
  )
}

function OrbitalRing({ radius, speed, tilt, color, offset = 0 }) {
  const ref = useRef()
  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.z += delta * speed
  })
  return (
    <group ref={ref} rotation={[tilt, offset, 0]}>
      <Torus args={[radius, 0.015, 8, 64]}>
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} transparent opacity={0.6} />
      </Torus>
    </group>
  )
}

function Node({ position, color, label, active, onClick }) {
  const ref = useRef()
  useFrame((state) => {
    if (ref.current) {
      ref.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 1.5 + position[0]) * 0.08
    }
  })
  return (
    <group position={position} ref={ref} onClick={onClick}>
      <Sphere args={[0.12, 16, 16]}>
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={active ? 1 : 0.4}
          roughness={0.3}
          metalness={0.7}
        />
      </Sphere>
      {active && (
        <Sphere args={[0.2, 16, 16]}>
          <meshBasicMaterial color={color} transparent opacity={0.2} />
        </Sphere>
      )}
    </group>
  )
}

function ConnectionLine({ start, end, color, active }) {
  const points = useMemo(() => {
    const s = new THREE.Vector3(...start)
    const e = new THREE.Vector3(...end)
    const mid = s.clone().lerp(e, 0.5)
    mid.y += 0.3
    const curve = new THREE.QuadraticBezierCurve3(s, mid, e)
    return curve.getPoints(20)
  }, [start, end])

  return (
    <Line
      points={points}
      color={color}
      lineWidth={active ? 2 : 0.5}
      transparent
      opacity={active ? 0.8 : 0.3}
    />
  )
}

function Scene({ nodes, selectedNode, onSelectNode, coreState }) {
  const nodePositions = nodes.map((n, i) => {
    const angle = (i / nodes.length) * Math.PI * 2
    const radius = 2.2
    return [Math.cos(angle) * radius, Math.sin(angle * 1.5) * 0.5, Math.sin(angle) * radius]
  })

  return (
    <>
      <ambientLight intensity={0.2} />
      <pointLight position={[0, 0, 0]} intensity={2} color="#00E5CC" distance={10} />
      <pointLight position={[5, 5, 5]} intensity={0.5} color="#3987FF" />
      <pointLight position={[-5, -5, -5]} intensity={0.3} color="#9B6DFF" />

      <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.5}>
        <CoreSphere state={coreState} />
      </Float>

      <OrbitalRing radius={1.5} speed={0.4} tilt={0.3} color="#00E5CC" />
      <OrbitalRing radius={1.9} speed={-0.3} tilt={-0.2} color="#3987FF" offset={1} />
      <OrbitalRing radius={2.3} speed={0.2} tilt={0.5} color="#9B6DFF" offset={2} />

      {nodes.map((node, i) => {
        const pos = nodePositions[i]
        const isActive = selectedNode === i
        return (
          <group key={i}>
            <ConnectionLine start={[0, 0, 0]} end={pos} color={node.color} active={isActive || coreState === 'running'} />
            <Node
              position={pos}
              color={node.color}
              label={node.label}
              active={isActive}
              onClick={() => onSelectNode(i)}
            />
          </group>
        )
      })}

      <OrbitControls
        enableZoom={false}
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.5}
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 1.8}
      />
    </>
  )
}

export default function AICore({ nodes = [], selectedNode, onSelectNode, coreState = 'idle' }) {
  return (
    <div className="w-full h-full">
      <Canvas camera={{ position: [0, 0, 6], fov: 50 }} dpr={[1, 2]}>
        <Suspense fallback={null}>
          <Scene nodes={nodes} selectedNode={selectedNode} onSelectNode={onSelectNode} coreState={coreState} />
        </Suspense>
      </Canvas>
    </div>
  )
}
