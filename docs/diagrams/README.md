# Diagrams

LLM-jp-Moshi / mstts のアーキテクチャを描いた drawio (`.drawio`) ファイル置き場。

## 一覧

| ファイル | 内容 |
|:---------|:-----|
| `llm_jp_moshi_mono.drawio`  | **Stage 1 mono** (1 text + 8 audio = 9 ch)。Moshi の最小構成 |
| `llm_jp_moshi_mstts.drawio` | **Stage 2/3 mstts** (1 text + 8 main + 8 other = 17 ch)。2 話者対話 TTS |
| `validate_drawio.py`        | XML 健全性チェッカー (0345 から拝借) |

両図のキーパラメータ:
- **Temporal Transformer** (≈ 7B params): dim=4096, L=32, H=32, RoPE / causal / ctx=3000
- **Depth Transformer (Depformer)**: dim=1024, L=6, dep_q=8, multi-linear, per-step weights
- **Mimi codec**: 24 kHz wav → 12.5 Hz, 8 codebooks (card=2048 each)
- **Text tokenizer**: rinna_gpt2 SentencePiece (vocab=32000)

## 開き方・編集

### Web (推奨)
1. [app.diagrams.net](https://app.diagrams.net/) を開く
2. "Open Existing Diagram" → "Device" から `.drawio` を選択
3. 編集後 "File → Save" → ローカルに上書き保存

### VS Code
"Draw.io Integration" 拡張機能 (`hediet.vscode-drawio`) をインストールすれば、
VS Code から直接編集可。

### 画像 (PNG/SVG/PDF) へのエクスポート

drawio CLI がインストールされていれば:
```bash
drawio --export --format png llm_jp_moshi_mstts.drawio
```

なければ web 版で `File → Export As → PNG/SVG/PDF` を選択。

## 編集ポリシー

- ソースとなる `.drawio` のみコミット (PNG/SVG エクスポート結果は基本的に commit しない)
- スライド資料に貼る目的でエクスポート済み画像が必要なら `docs/images/` 配下に置く

## バリデーション (`validate_drawio.py`)

drawio で開いて初めて発覚する render エラー (`d.setId is not a function`
など) の **多くは XML 構造の問題**。コミット前に静的チェックを推奨:

```bash
python3 docs/diagrams/validate_drawio.py docs/diagrams/llm_jp_moshi_mstts.drawio
python3 docs/diagrams/validate_drawio.py docs/diagrams/llm_jp_moshi_mono.drawio
```

チェック項目:
1. XML パース可能か
2. 重複 ID
3. ID の特殊文字 (DOM-safe か)
4. **ID が JS prototype メソッド名と衝突しないか** (後述の落とし穴)
5. `source` / `target` / `parent` 参照先の存在
6. edge の source/target 不在 (sourcePoint/targetPoint のみは許容)
7. `vertex=1` と `edge=1` の同時指定
8. 親不在 (root 以外)
9. edge の child cell が edgeLabel 規約を満たすか
10. style 文字列の `key=value;` 整形
11. parent の循環参照

完全な保証ではない (drawio 内部の JS でしか検出できないエラーもある) が、
過去にハマった `d.setId is not a function` は本ツールでカバーする項目で
発生していた。

完全に確証が欲しい場合は drawio Desktop CLI (`drawio --export`) で
書き出してみるのが確実。

---

## 落とし穴: `mxCell id` に JS prototype メソッド名を使わない

### 症状
drawio でファイルを開くと以下のエラーでレンダリングできない:
```
EditorUi.handleError: TypeError: d.setId is not a function
    at mxObjectCodec.beforeDecode (app.min.js:...)
    at mxCodec.decodeCell (...)
    ...
```

### 原因
drawio は cell を JS の plain object `{}` に **ID をキーとして** 格納する。
ID が `concat` の場合、`cellMap["concat"]` は **`Array.prototype.concat`
(関数オブジェクト) にヒット** してしまう。decoder は「存在しない」と判断
して新規 cell を作るべきところを、この関数を cell と誤認し、その関数に
対して `setId(...)` を呼ぼうとして `setId is not a function` で死ぬ。

### 禁止すべき ID 名 (代表例)

**絶対に使うな** (`Object` / `Array` / `String` / `Function` の prototype
メソッド名と衝突):

```
concat  length  constructor  toString  valueOf  hasOwnProperty
push  pop  shift  unshift  slice  splice  map  filter  reduce
forEach  find  indexOf  includes  join  reverse  sort  every
some  keys  values  entries  flat  flatMap  at
charAt  replace  search  split  trim  startsWith  endsWith
apply  bind  call  name
__proto__  __defineGetter__  ...
```

完全なリストは `validate_drawio.py` の `js_proto_names` セットを参照。

### 推奨ネーミング

- **接尾辞を付ける**: `concat` → `concat_layer`, `push` → `push_stack`
- **プレフィックス**: `n_` / `node_` / `v_` (vertex) / `e_` (edge)
- **ランダム短縮形**: `n01`, `n02`, ... (drawio Desktop がエクスポートする
  ネイティブ形式もこの種)

### 実体験ログ (2026-04-22 @ 0345)

- `4stream_structure.drawio` で cell id `concat` を使ったらレンダーエラー
- bisect で `concat` が原因と特定
- 修正: `concat` → `concat_layer` にリネーム
- 再発防止: validator に denylist 追加 (項目 #4)
