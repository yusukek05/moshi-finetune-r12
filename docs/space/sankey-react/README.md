# LLM-jp-Moshi データフロー Sankey (React + ECharts)

External-facing version of the LLM-jp-Moshi data lineage Sankey, published
to the public Hugging Face Space:

→ https://huggingface.co/spaces/abePclWaseda/llm-jp-moshi-data-sankey

Built with **Vite + React 19 + TypeScript** wrapping **ECharts** for the
Sankey itself. Features:

- Filter chips: 全部 / v1 系 / mstts 系 / 商用クリーンのみ
- Click node / edge → right-side detail panel with full description
- Hover tooltip with exact hour values (the rendered edge width uses
  √-scaling so small flows stay visible next to J-CHAT's 72k h)
- Responsive layout (mobile: chart + panel stack vertically)
- License color coding: 商用可 / NC ⚠ / 予定 / ベース / その他

## Data model

`src/data.ts` is the single source of truth — `NODES` + `EDGES` arrays
with TypeScript types. To update numbers (e.g. after a new corpus is
added or measured), edit this file, rebuild, and re-upload.

The Python counterpart `tools/build_sankey_overview.py` (in the repo
root) renders the same dataset to a Plotly HTML and is kept around as a
fallback / for the per-corpus measurement code.

## Local development

```bash
cd docs/space/sankey-react
npm install
npm run dev   # vite dev server (hot reload)
npm run build # → dist/
```

## Publishing to the HF Space

```bash
cd docs/space/sankey-react
npm run build

# Upload dist/* to the Space (from repo root)
cd ../../..
module load python/3.12/3.12.9
uv run --no-project --with huggingface_hub python -c "
from huggingface_hub import upload_folder
upload_folder(
    folder_path='docs/space/sankey-react/dist',
    repo_id='abePclWaseda/llm-jp-moshi-data-sankey',
    repo_type='space',
    commit_message='Update data flow Sankey')"
```

The previous Plotly version is preserved at `/index_plotly.html` in the
Space repo for reference.
