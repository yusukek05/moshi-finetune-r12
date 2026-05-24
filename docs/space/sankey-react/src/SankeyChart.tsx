import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  NODES,
  EDGES,
  PALETTE,
  type NodeDef,
  type EdgeDef,
  type Category,
} from './data'

type EChartsClickParams = {
  dataType: 'node' | 'edge'
  data: NodeData | EdgeData
}

type SelectedDetail =
  | { kind: 'node'; node: NodeDef }
  | { kind: 'edge'; edge: EdgeDef }

interface NodeData {
  name: string
  itemStyle: { color: string; borderColor: string; borderWidth: number }
  _ref: NodeDef
}

interface EdgeData {
  source: string
  target: string
  value: number
  _ref: EdgeDef
  lineStyle: { color: string; curveness: number }
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  return `rgba(${r},${g},${b},${alpha})`
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] ?? c),
  )
}

interface SankeyChartProps {
  filter: Category | 'all'
  onSelect: (detail: SelectedDetail | null) => void
}

export function SankeyChart({ filter, onSelect }: SankeyChartProps) {
  const option = useMemo<EChartsOption>(() => {
    const visibleNodes: NodeDef[] =
      filter === 'all' ? NODES : NODES.filter((n) => n.cats.includes(filter))
    const visibleIds = new Set(visibleNodes.map((n) => n.id))
    const visibleEdges: EdgeDef[] = EDGES.filter(
      (e) =>
        visibleIds.has(e.source) &&
        visibleIds.has(e.target) &&
        (filter === 'all' || e.cats.includes(filter)),
    )

    const nodeNameById = new Map(visibleNodes.map((n) => [n.id, n.label]))

    const nodes: NodeData[] = visibleNodes.map((n) => ({
      name: n.label,
      itemStyle: {
        color: PALETTE[n.license],
        borderColor: '#222',
        borderWidth: 0.6,
      },
      _ref: n,
    }))

    const links: EdgeData[] = visibleEdges.map((e) => ({
      source: nodeNameById.get(e.source) ?? e.source,
      target: nodeNameById.get(e.target) ?? e.target,
      value: Math.max(1, Math.sqrt(e.hours)),
      _ref: e,
      lineStyle: {
        color: hexToRgba(PALETTE[e.license], 0.55),
        curveness: 0.5,
      },
    }))

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        backgroundColor: 'rgba(30,30,30,0.95)',
        borderColor: '#222',
        textStyle: { color: '#fff', fontSize: 12 },
        extraCssText: 'max-width: 320px; white-space: normal;',
        formatter: (params: unknown) => {
          const p = params as EChartsClickParams
          if (p.dataType === 'node') {
            const n = (p.data as NodeData)._ref
            return (
              `<b>${escapeHtml(n.label)}</b><br/>` +
              `<span style="font-size:11px;color:#bbb">${escapeHtml(n.hover)}</span>`
            )
          }
          if (p.dataType === 'edge') {
            const e = (p.data as EdgeData)._ref
            return (
              `<b>${escapeHtml(e.source)} → ${escapeHtml(e.target)}</b><br/>` +
              `${escapeHtml(e.label)}<br/>` +
              `<span style="font-size:11px;color:#bbb">実値: ${e.hours.toLocaleString()} h</span>`
            )
          }
          return ''
        },
      },
      series: [
        {
          type: 'sankey',
          nodeAlign: 'justify',
          layoutIterations: 64,
          nodeWidth: 14,
          nodeGap: 14,
          data: nodes,
          links: links,
          label: {
            color: '#222',
            fontSize: 12,
            fontFamily:
              'Inter, -apple-system, system-ui, "Hiragino Sans", "Yu Gothic", "Noto Sans JP", sans-serif',
          },
          emphasis: { focus: 'adjacency' },
          left: 10,
          right: 180,
          top: 10,
          bottom: 10,
        },
      ],
    }
  }, [filter])

  return (
    <ReactECharts
      option={option}
      style={{ width: '100%', height: '100%' }}
      notMerge={true}
      lazyUpdate={false}
      onEvents={{
        click: (params: unknown) => {
          const p = params as EChartsClickParams
          if (p.dataType === 'node') {
            onSelect({ kind: 'node', node: (p.data as NodeData)._ref })
          } else if (p.dataType === 'edge') {
            onSelect({ kind: 'edge', edge: (p.data as EdgeData)._ref })
          }
        },
      }}
    />
  )
}

export type { SelectedDetail }
