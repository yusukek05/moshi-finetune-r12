# 音声途切れ（ぶつぶつ）の原因分析

## audio-processor.ts で考えられる原因

### 1. **バッファアンダーラン時の再生停止**（主因）
- **箇所**: 166–175行
- `out_idx < output.length` のとき（出力バッファを埋めきれない＝フレームが足りない）、`resetStart()` で再生を止め、`initialBufferSamples` が溜まるまで待つ。
- その間は音声が止まるため、**「ぶつぶつ」の直接原因**になる。

### 2. **バッファが小さい**
- `initialBufferSamples` = 80ms、`partialBufferSamples` = 10ms、`maxBufferSamples` = 10ms
- 合計で約 100ms しか溜められず、ネットワークやメインスレッドの揺らぎですぐアンダーランしやすい。

### 3. **Worklet 内の console.log**
- 45–46, 49–50, 63–66, 145, 167, 171 行などで `console.log` を実行している。
- Audio Worklet はレンダリング/音声用のスレッドで動くため、同期的なログが処理をブロックし、**再生のジッターや途切れ**の原因になりうる。

### 4. **フレーム受信ごとの postMessage**
- 39–75 行で、フレームを受信するたびに `this.port.postMessage(...)` でメインスレッドに統計を送っている。
- 頻度が高いとメインスレッドの負荷が増え、その結果「デコード済みフレームを worklet に渡す」処理が遅れ、アンダーランにつながる可能性がある。

---

## useServerAudio.ts で考えられる原因

### 1. **デコード済み音声がメインスレッド経由**
- デコード: DecoderWorker → `onWorkerMessage`（メインスレッド）→ `onDecode` → `worklet.port.postMessage`。
- **すべてのフレームがメインスレッドを経由**するため、React の描画や WaveformVisualizer の canvas などでメインスレッドが忙しいと、worklet に渡すのが遅れ、上記のアンダーランを招く。

### 2. **postMessage で ArrayBuffer を転送していない**
- 47 行: `worklet.current.port.postMessage({ frame: data, ... })` で `data`（Float32Array）をそのまま渡している。
- 転送リストを使わないと、メインスレッドでバッファのコピーが発生し、負荷と遅延が増える。`[data.buffer]` で転送すると負荷軽減の余地がある。

### 3. **デバッグ用 console.log**
- 85–86, 115, 129, 148 行などのログが、メッセージ処理のタイミングでメインスレッドを占有し、音声パイプラインを遅らせる可能性がある。

---

## 推奨する修正の方向性（実施済み）

1. **audio-processor.ts** ✅
   - Worklet 内の `console.log` を削除した（音声スレッドのブロックを防止）。
   - 統計用の `port.postMessage` を 10 フレームに 1 回に間引いた（メインスレッド負荷軽減）。
   - `statsCounter` を導入し、`initState` で初期化。

2. **useServerAudio.ts** ✅
   - `worklet.port.postMessage(..., [data.buffer])` で転送リストを使用（コピー削減）。
   - デバッグ用の `console.log` を削除。

3. **まだ試していない場合**
   - 初期バッファを増やす（`initialBufferSamples` や `maxBufferSamples`）でアンダーランをさらに減らせる可能性あり。
   - メインスレッド負荷を下げる（波形描画の軽量化など）。

---

## 「Dropping packets」が出る理由（バッファ溢れの向き）

### 条件
- `currentSamples() >= totalMaxBufferSamples()` のときに **古いパケットを捨てる**（レイテンシ抑制のため）。

### 意味
- Worklet 内部の **再生待ちバッファ** が **上限を超えた** ＝ **届いた量が、再生で消費する量を（一時的に）上回った** 状態。
- 再生は `process()` で **リアルタイム（サンプルレート通り）** で消費されるので、「再生が遅い」のではなく、**「届くタイミングがバーストしている」** と考えられる。

### 想定される流れ
1. **WebSocket → メインスレッド**: サーバーからの音声が一気に届く、またはメインスレッドが忙しくて `onmessage` が溜まる。
2. **メインスレッド → DecoderWorker**: デコード要求がまとまって処理される。
3. **DecoderWorker → メインスレッド → Worklet**: デコード結果がまとまったタイミングで `postMessage` され、Worklet に短時間に多数のフレームが届く。
4. **Worklet**: 届いたフレームを `this.frames` に積む。再生は一定レートなので、一時的に「入り > 出」となりバッファが増加。
5. **currentSamples() >= totalMaxBufferSamples()**: 上限超過 → **Dropping packets** で古いパケットを捨てる。

### 結論
- **「AudioWorklet の送受・処理が間に合わず」** というより、  
  **「上流（WebSocket／メインスレッド／デコーダ）から Worklet への供給がバーストし、Worklet 側バッファが一気に溜まって溢れている」** と考えられる。
- Worklet はリアルタイムで消費しているので、溢れの主因は **供給側のバースト** であり、Worklet の処理速度不足ではない。

---

## ログ「Dropping packets」「Packet dropped」「Increased maxBuffer」が出る理由

### ログの意味
- **Dropping packets '229.3' '200.0'**  
  再生待ちバッファが 229ms になり、上限 200ms を超えたため、古いパケットを捨てている。
- **Packet dropped '120.0'**  
  ドロップ後、バッファが約 120ms まで減った。
- **Increased maxBuffer to 80.0**  
  次回から溢れにくくするため、`maxBufferSamples` を 80ms に増やした。

### 音声がぶつぶつする理由
1. **ドロップ**  
   バッファ溢れで古いパケットを捨てているため、その分の音声が欠落する。
2. **アンダーラン**  
   逆にバッファが空になると「Missed some audio」となり、再生を止めてバッファが溜まるまで待つため、その間も途切れる。

### 対策（実施済み・推奨）
1. **クライアントの再ビルド**  
   `./build_client.sh` で dist を更新する。古い worklet（`audio-processor-xxx.js`）には上記ログが残っているため、再ビルドでログ削除版に変わる。
2. **バッファの余裕を増やす**  
   `partialBufferSamples` を 10ms→20ms、`maxBufferSamples` を 10ms→30ms に変更済み。バーストを吸収しやすくなり、ドロップが減る（レイテンシはやや増える）。
3. **Worklet 内の console.log 削除**  
   音声スレッドの負荷を減らすため、ソース側のログは削除済み。再ビルド後に反映される。
