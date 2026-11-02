import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import dagre from '@dagrejs/dagre'
import {
  Background,
  BackgroundVariant,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useNodesInitialized,
  useNodesState,
  useReactFlow,
  useViewport,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import { flowGraphs, type FlowGraph, type FlowGraphId, type FlowNodeKind } from './data'
import '@xyflow/react/dist/style.css'
import './styles.css'

type DiagramNode = Node<{ label: string; kind: FlowNodeKind }, 'step'>

function StepNode({ data }: NodeProps<DiagramNode>) {
  return <div className={`flow-step flow-step--${data.kind}`}>
    <Handle type="target" position={Position.Top} isConnectable={false} />
    <span>{data.label}</span>
    <Handle type="source" position={Position.Bottom} isConnectable={false} />
  </div>
}

const nodeTypes = { step: StepNode }

function wrapLabel(label: string) {
  return label.split('\n').flatMap(line => {
    const lines: string[] = []
    for (const word of line.split(' ')) {
      const last = lines.length - 1
      if (last < 0 || lines[last].length + word.length + 1 > 22) lines.push(word)
      else lines[last] += ` ${word}`
    }
    return lines
  }).join('\n')
}

function layout(graph: FlowGraph) {
  const geometry = new dagre.graphlib.Graph({ multigraph: true })
  geometry.setDefaultEdgeLabel(() => ({}))
  geometry.setGraph({ rankdir: 'TB', nodesep: 42, ranksep: 60, marginx: 30, marginy: 30 })
  const labels = new Map(graph.nodes.map(node => [node.id, wrapLabel(node.label)]))
  const sizes = new Map(graph.nodes.map(node => [node.id, {
    width: 236,
    height: Math.max(76, labels.get(node.id)!.split('\n').length * 22 + 30),
  }]))
  graph.nodes.forEach(node => geometry.setNode(node.id, sizes.get(node.id)!))
  graph.edges.forEach(edge => geometry.setEdge(edge.source, edge.target, {}, edge.id))
  dagre.layout(geometry)
  const nodes: DiagramNode[] = graph.nodes.map(node => {
    const position = geometry.node(node.id)
    const size = sizes.get(node.id)!
    return {
      id: node.id,
      type: 'step',
      data: { label: labels.get(node.id)!, kind: node.kind },
      position: { x: position.x - size.width / 2, y: position.y - size.height / 2 },
      style: { width: size.width, height: size.height },
      ariaLabel: `${node.label.replaceAll('\n', '. ')}. Use as setas para mover este bloco.`,
    }
  })
  const edges: Edge[] = graph.edges.map(edge => ({
    ...edge,
    type: 'smoothstep',
    markerEnd: { type: MarkerType.ArrowClosed, color: '#193341', width: 18, height: 18 },
    style: edge.dashed ? { strokeDasharray: '5 4' } : undefined,
    labelBgPadding: [6, 3],
    labelBgBorderRadius: 3,
    ariaLabel: `${edge.source} para ${edge.target}${edge.label ? `: ${edge.label}` : ''}`,
    focusable: false,
    selectable: false,
    deletable: false,
  }))
  return { nodes, edges, width: geometry.graph().width ?? 800, height: geometry.graph().height ?? 600 }
}

function Diagram({ graph, host }: { graph: FlowGraph; host: HTMLElement }) {
  const initial = useMemo(() => layout(graph), [graph])
  const [nodes, setNodes, onNodesChange] = useNodesState<DiagramNode>(initial.nodes)
  const [expanded, setExpanded] = useState(false)
  const shell = useRef<HTMLDivElement>(null)
  const initialViewApplied = useRef(false)
  const nodesInitialized = useNodesInitialized()
  const { fitView, setViewport, getViewport, zoomIn, zoomOut } = useReactFlow<DiagramNode>()
  const { zoom } = useViewport()
  const shortMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 180

  const fit = useCallback(() => {
    void fitView({ padding: 0.14, minZoom: 0.12, maxZoom: 1.2, duration: shortMotion() })
  }, [fitView])

  // A dialog moves the existing host, so dragging and zoom survive expansion.
  // Native modal behavior supplies focus trapping, Escape and an inert backdrop.
  useEffect(() => {
    if (!expanded) return
    const placeholder = document.createComment('flow-canvas')
    const dialog = document.createElement('dialog')
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    dialog.className = 'flow-dialog'
    dialog.setAttribute('aria-label', graph.title)
    host.before(placeholder)
    document.body.append(dialog)
    dialog.append(host)
    const closed = () => setExpanded(false)
    dialog.addEventListener('close', closed)
    const previousViewport = getViewport()
    dialog.showModal()
    let secondFrame = 0
    const firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(fit)
    })
    return () => {
      window.cancelAnimationFrame(firstFrame)
      window.cancelAnimationFrame(secondFrame)
      dialog.removeEventListener('close', closed)
      dialog.close()
      placeholder.replaceWith(host)
      dialog.remove()
      previousFocus?.focus({ preventScroll: true })
      window.requestAnimationFrame(() => void setViewport(previousViewport))
    }
  }, [expanded, graph.title, host, fit, getViewport, setViewport])

  const initialView = useCallback(() => {
    const bounds = shell.current?.querySelector('.flow-stage')?.getBoundingClientRect()
    if (!bounds) return
    if (initial.height > bounds.height * 2) {
      // Long event cycles start at the beginning rather than as unreadable thumbnails.
      const scale = Math.min(0.8, Math.max(0.5, (bounds.width - 40) / initial.width))
      const first = initial.nodes[0]
      void setViewport({ x: bounds.width / 2 - (first.position.x + 118) * scale, y: 25, zoom: scale })
    } else {
      void fitView({ padding: 0.1, minZoom: 0.35, maxZoom: 1 })
    }
  }, [initial, setViewport, fitView])

  useEffect(() => {
    if (nodesInitialized && !initialViewApplied.current) {
      initialViewApplied.current = true
      initialView()
    }
  }, [nodesInitialized, initialView])

  const reset = () => {
    setNodes(initial.nodes.map(node => ({ ...node, position: { ...node.position }, selected: false })))
    window.requestAnimationFrame(fit)
  }

  return <div className="flow-shell" ref={shell}>
    <div className="flow-toolbar" aria-label={`Controles: ${graph.title}`}>
      <div className="flow-zoom">
        <button type="button" aria-label="Diminuir zoom" title="Diminuir zoom" onClick={() => void zoomOut({ duration: shortMotion() })}>−</button>
        <output aria-label="Nível de zoom">{Math.round(zoom * 100)}%</output>
        <button type="button" aria-label="Aumentar zoom" title="Aumentar zoom" onClick={() => void zoomIn({ duration: shortMotion() })}>+</button>
      </div>
      <button type="button" onClick={fit} title="Enquadrar o fluxograma inteiro">Ajustar</button>
      <button type="button" onClick={reset} title="Restaurar a posição dos blocos">Reorganizar</button>
      <button className="flow-expand" type="button" onClick={() => setExpanded(!expanded)} aria-expanded={expanded}>
        {expanded ? 'Fechar tela cheia' : 'Tela cheia'}
      </button>
    </div>
    <div className="flow-stage" role="region" aria-label={graph.title}>
      <ReactFlow<DiagramNode>
        id={`fluxo-${graph.id}`}
        nodes={nodes}
        edges={initial.edges}
        onNodesChange={onNodesChange}
        nodeTypes={nodeTypes}
        nodesConnectable={false}
        edgesReconnectable={false}
        deleteKeyCode={null}
        minZoom={0.12}
        maxZoom={2}
        zoomOnScroll={false}
        zoomActivationKeyCode={['Control', 'Meta']}
        zoomOnPinch
        preventScrolling={false}
        panOnDrag
        attributionPosition="bottom-left"
        ariaLabelConfig={{
          'node.a11yDescription.default': 'Pressione Enter para selecionar o bloco. Use as setas para movê-lo. Escape cancela a seleção.',
          'node.a11yDescription.keyboardDisabled': 'Pressione Enter para selecionar o bloco.',
          'node.a11yDescription.ariaLiveMessage': ({ direction, x, y }) => `Bloco movido para ${direction}. Posição ${Math.round(x)}, ${Math.round(y)}.`,
          'minimap.ariaLabel': 'Visão geral do fluxograma. Arraste para navegar.',
        }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#CBD6DD" />
        <MiniMap pannable zoomable position="bottom-right" nodeColor={node => node.data.kind === 'model' ? '#2452D6' : '#CBD6DD'} />
      </ReactFlow>
    </div>
    <p className="flow-instructions">Arraste os blocos ou o fundo. Use + / − para ampliar. No celular, use dois dedos para o zoom.</p>
  </div>
}

document.querySelectorAll<HTMLElement>('.flow-canvas[data-flow]').forEach(host => {
  const id = host.dataset.flow as FlowGraphId
  const graph = flowGraphs[id]
  if (!graph) return
  createRoot(host).render(<ReactFlowProvider><Diagram graph={graph} host={host} /></ReactFlowProvider>)
})
