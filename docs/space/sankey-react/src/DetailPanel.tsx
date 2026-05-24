import { GROUP_LABEL, LICENSE_LABEL, PALETTE } from './data'
import type { SelectedDetail } from './SankeyChart'

interface DetailPanelProps {
  detail: SelectedDetail | null
}

export function DetailPanel({ detail }: DetailPanelProps) {
  if (!detail) {
    return (
      <aside className="detail">
        <h2>詳細</h2>
        <p className="placeholder">
          node / edge をクリックすると、ここに詳細が表示されます。
          <br />
          hover で実値（hour 数）が tooltip に出ます。
        </p>
      </aside>
    )
  }

  if (detail.kind === 'node') {
    const n = detail.node
    return (
      <aside className="detail">
        <h2>
          NODE · {GROUP_LABEL[n.group]}{' '}
          <span className="badge" style={{ background: PALETTE[n.license] }}>
            {LICENSE_LABEL[n.license]}
          </span>
        </h2>
        <div className="title">{n.label}</div>
        <div className="desc">{n.hover}</div>
        <div className="meta">
          所属カテゴリ: {n.cats.map(catLabel).join(' / ')}
        </div>
      </aside>
    )
  }

  const e = detail.edge
  return (
    <aside className="detail">
      <h2>
        EDGE{' '}
        <span className="badge" style={{ background: PALETTE[e.license] }}>
          {LICENSE_LABEL[e.license]}
        </span>
      </h2>
      <div className="title">
        {e.source} → {e.target}
      </div>
      <div className="desc">{e.label}</div>
      <div className="meta">
        実音声時間: <b>{e.hours.toLocaleString()} h</b>
      </div>
    </aside>
  )
}

function catLabel(c: string): string {
  return (
    { v1: 'v1 系', mstts: 'mstts 系', clean: '商用クリーン', shared: '共通' }[c] ??
    c
  )
}
