import { useState } from 'react'
import './App.css'
import { SankeyChart, type SelectedDetail } from './SankeyChart'
import { DetailPanel } from './DetailPanel'
import { PALETTE, SNAPSHOT, type Category } from './data'

type FilterValue = Category | 'all'

const FILTERS: { value: FilterValue; label: string }[] = [
  { value: 'all', label: '全部' },
  { value: 'v1', label: 'v1 系（対話 LM）' },
  { value: 'mstts', label: 'mstts 系（音声合成）' },
  { value: 'clean', label: '商用クリーンのみ' },
]

function App() {
  const [filter, setFilter] = useState<FilterValue>('all')
  const [detail, setDetail] = useState<SelectedDetail | null>(null)

  return (
    <div className="page">
      <header className="page-header">
        <h1>LLM-jp-Moshi データフロー</h1>
        <p>
          日本語対話音声モデル LLM-jp-Moshi シリーズの学習データ・中間生成物・モデル系譜。
          edge 幅は √(音声時間) で圧縮表示、ホバーで実値表示。クリックで詳細パネル。
        </p>
      </header>

      <nav className="controls">
        <div className="chips">
          <span className="label">フィルタ:</span>
          {FILTERS.map((f) => (
            <button
              key={f.value}
              type="button"
              className={`chip${filter === f.value ? ' active' : ''}`}
              onClick={() => {
                setFilter(f.value)
                setDetail(null)
              }}
            >
              {f.label}
            </button>
          ))}
        </div>
        <div className="legend">
          <span className="label">ライセンス:</span>
          {(
            [
              ['ok', '商用可'],
              ['nc', '非商用 ⚠'],
              ['plan', '予定'],
              ['base', 'ベース'],
              ['unk', 'その他'],
            ] as const
          ).map(([key, label]) => (
            <span key={key} className="legend-item">
              <span className="legend-swatch" style={{ background: PALETTE[key] }} />
              {label}
            </span>
          ))}
        </div>
      </nav>

      <main className="main">
        <div className="chart-wrap">
          <SankeyChart filter={filter} onSelect={setDetail} />
        </div>
        <DetailPanel detail={detail} />
      </main>

      <footer className="page-footer">
        Snapshot: {SNAPSHOT} · データ系譜は abe@pcl.cs.waseda.ac.jp 管理 ·{' '}
        <a
          href="https://github.com/abePclWaseda/moshi-finetune"
          target="_blank"
          rel="noopener noreferrer"
        >
          source
        </a>
      </footer>
    </div>
  )
}

export default App
