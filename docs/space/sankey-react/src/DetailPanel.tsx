import { GROUP_LABEL, LICENSE_LABEL, PALETTE } from './data'
import type { NodeDef } from './data'
import type { SelectedDetail } from './SankeyChart'

interface DetailPanelProps {
  detail: SelectedDetail | null
}

export function DetailPanel({ detail }: DetailPanelProps) {
  if (!detail) {
    return (
      <aside className="detail">
        <h2>この図の読み方</h2>
        <div className="guide">
          <p className="guide-line">
            <b>横方向</b>：左 = <span className="kbd">コーパス</span> / 中 ={' '}
            <span className="kbd">中間生成物</span> / 右 ={' '}
            <span className="kbd">モデル</span>
          </p>
          <p className="guide-line">
            <b>色</b>：
            <span className="legend-swatch" style={{ background: '#3a8540' }} /> 商用可 ／
            <span className="legend-swatch" style={{ background: '#c64a3b' }} /> 非商用 (NC) ⚠ ／
            <span className="legend-swatch" style={{ background: '#cba135' }} /> 予定 ／
            <span className="legend-swatch" style={{ background: '#2f6fb7' }} /> ベース
          </p>
          <p className="guide-line">
            <b>帯の太さ</b>：√(音声時間) で圧縮。
            72k h と 12 h を同じ画面に共存させるため。
            hover で実際の時間が出ます。
          </p>
          <p className="guide-line">
            <b>左上フィルタ</b>：v1 系 (対話 LM) / mstts 系 (音声合成) /
            商用クリーンのみ — を切り替え可。初期表示は{' '}
            <i>商用クリーンのみ</i>。
          </p>
          <p className="guide-line guide-hint">
            ↓ node / edge をクリックすると、ここに詳細 + 音声サンプル (一部) が出ます。
          </p>
        </div>
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
        <SamplesSection node={n} />
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

function SamplesSection({ node }: { node: NodeDef }) {
  const hasSamples = node.samples && node.samples.length > 0
  const hasLinks = node.externalLinks && node.externalLinks.length > 0
  if (!hasSamples && !hasLinks) return null

  return (
    <div className="samples">
      <h3>音声サンプル</h3>
      {node.license === 'nc' && hasSamples && (
        <p className="no-sample">
          ⚠️ 非商用 (NC) ライセンスのコーパスです。試聴は研究目的の参考用として提供しています。
        </p>
      )}
      {node.samples?.map((s, i) => (
        <div key={i} className="sample">
          {s.caption && <div className="sample-caption">{s.caption}</div>}
          {s.topic && <div className="sample-topic">「{s.topic}」</div>}
          <audio
            controls
            preload="none"
            src={`${import.meta.env.BASE_URL}samples/${node.id}/${s.src}`}
          />
          {(s.duration || s.license) && (
            <div className="sample-meta">
              {s.duration ? <span>{s.duration}s</span> : null}
              {s.duration && s.license ? <span> · </span> : null}
              {s.license ? <span>{s.license}</span> : null}
            </div>
          )}
          {s.source && <div className="sample-source">出典: {s.source}</div>}
        </div>
      ))}
      {node.externalLinks?.map((l, i) => (
        <div key={`link-${i}`} className="sample-external">
          🔗{' '}
          <a href={l.href} target="_blank" rel="noopener noreferrer">
            {l.label}
          </a>
        </div>
      ))}
    </div>
  )
}

function catLabel(c: string): string {
  return (
    { v1: 'v1 系', mstts: 'mstts 系', clean: '商用クリーン', shared: '共通' }[c] ??
    c
  )
}
