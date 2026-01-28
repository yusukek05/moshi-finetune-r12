import { FC, useEffect, useRef, useState } from "react";
import { useServerText } from "../../hooks/useServerText";

type WaveformVisualizerProps = {
  aiAnalyser: AnalyserNode | null;
  humanAnalyser: AnalyserNode | null;
  durationSec: number;
  currentTimeSec: number;
  displayColor?: boolean;
};

const WIDTH = 1920;
const HEIGHT = 1080;
const FPS = 30;
const ENVELOPE_SAMPLES = 500; // エンベロープのサンプル数

// ガウシアン平滑化（簡易版：移動平均）
const smoothEnvelope = (data: number[], windowSize: number = 5): number[] => {
  const smoothed = new Array(data.length);
  const halfWindow = Math.floor(windowSize / 2);
  
  for (let i = 0; i < data.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - halfWindow); j <= Math.min(data.length - 1, i + halfWindow); j++) {
      sum += data[j];
      count++;
    }
    smoothed[i] = sum / count;
  }
  return smoothed;
};

// AnalyserNodeから現在の振幅を取得
const getCurrentAmplitude = (analyser: AnalyserNode | null): number => {
  if (!analyser) {
    return 0;
  }
  
  const bufferLength = analyser.fftSize || 2048;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteTimeDomainData(dataArray);
  
  // RMSを計算（時間領域データから）
  let sum = 0;
  for (let i = 0; i < bufferLength; i++) {
    const normalized = (dataArray[i] - 128) / 128; // -1 to 1
    sum += normalized * normalized;
  }
  const rms = Math.sqrt(sum / bufferLength);
  
  return Math.min(Math.max(rms, 0), 1); // 0〜1にクランプ
};

export const WaveformVisualizer: FC<WaveformVisualizerProps> = ({
  aiAnalyser,
  humanAnalyser,
  durationSec,
  currentTimeSec,
  displayColor = false,
}) => {
  // SocketContext内でuseServerTextを呼び出す
  const { text: aiText, textColor: aiTextColor } = useServerText();
  
  // 各トークンの生成タイミングと色情報を追跡
  const tokenTimingsRef = useRef<{ token: string; time: number; color?: number; index: number }[]>([]);
  const lastTextLengthRef = useRef(0);
  
  // テキストが更新されたら、新しいトークンのタイミングを記録
  // トークンのタイミングは、durationSecに対する相対位置（0〜1）で記録
  useEffect(() => {
    if (aiText && aiText.length > lastTextLengthRef.current) {
      // 新しいトークンが追加された
      const newTokens = aiText.slice(lastTextLengthRef.current);
      // 現在の再生位置を0〜1の相対位置に変換
      const currentProgress = durationSec > 0 ? Math.min(Math.max((currentTimeSec || 0) / durationSec, 0), 1) : 0;
      
      newTokens.forEach((token, i) => {
        const globalIndex = lastTextLengthRef.current + i;
        // トークンが生成された時点での進行度を記録
        // 簡易的に、現在の進行度を使用（実際には各トークンの生成タイミングを追跡する必要がある）
        tokenTimingsRef.current.push({
          token,
          time: currentProgress, // 0〜1の相対位置
          color: aiTextColor && aiTextColor.length > globalIndex ? aiTextColor[globalIndex] : undefined,
          index: globalIndex,
        });
      });
      
      lastTextLengthRef.current = aiText.length;
      
      console.log("WaveformVisualizer: New tokens added", {
        newTokens,
        currentProgress,
        currentTimeSec,
        durationSec,
        totalTokens: tokenTimingsRef.current.length,
      });
    } else if (aiText && aiText.length < lastTextLengthRef.current) {
      // テキストがリセットされた（新しい会話開始）
      tokenTimingsRef.current = [];
      lastTextLengthRef.current = 0;
    }
  }, [aiText, aiTextColor, currentTimeSec, durationSec]);
  
  // デバッグ用：テキストの状態をログ出力
  useEffect(() => {
    if (aiText && aiText.length > 0) {
      console.log("WaveformVisualizer: AI text updated", {
        textLength: aiText.length,
        textArray: aiText,
        fullText: aiText.join(""),
        tokenTimings: tokenTimingsRef.current.length,
      });
    }
  }, [aiText]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [aiEnvelope, setAiEnvelope] = useState<number[]>(new Array(ENVELOPE_SAMPLES).fill(0));
  const [humanEnvelope, setHumanEnvelope] = useState<number[]>(new Array(ENVELOPE_SAMPLES).fill(0));
  const envelopeHistoryRef = useRef<{ ai: number[]; human: number[] }>({
    ai: [],
    human: [],
  });

  // デバッグ用：analyserの状態をログ出力（初回のみ）
  const hasLoggedRef = useRef(false);
  useEffect(() => {
    if ((aiAnalyser || humanAnalyser) && !hasLoggedRef.current) {
      console.log("WaveformVisualizer: analysers available", {
        ai: !!aiAnalyser,
        human: !!humanAnalyser,
        durationSec,
        currentTimeSec,
      });
      hasLoggedRef.current = true;
    }
  }, [aiAnalyser, humanAnalyser, durationSec, currentTimeSec]);

  // エンベロープを定期的に更新
  useEffect(() => {
    if (!aiAnalyser && !humanAnalyser) {
      // analyserがない場合は、履歴をクリア
      envelopeHistoryRef.current.ai = [];
      envelopeHistoryRef.current.human = [];
      setAiEnvelope(new Array(ENVELOPE_SAMPLES).fill(0));
      setHumanEnvelope(new Array(ENVELOPE_SAMPLES).fill(0));
      return;
    }

    // 更新間隔を調整（30fps → 60fpsでより滑らかに）
    const updateInterval = 1000 / 60; // 約16.67ms
    let frameCount = 0;

    const interval = setInterval(() => {
      frameCount++;
      
      // AIチャンネル
      if (aiAnalyser) {
        const amplitude = getCurrentAmplitude(aiAnalyser);
        
        // 振幅が閾値以上の時だけ追加（ノイズを除外）
        if (amplitude > 0.01) {
          // 新しいサンプルを追加（右側に追加）
          envelopeHistoryRef.current.ai.push(amplitude);
        } else {
          // 音声がない時も0を追加して時間軸を維持
          envelopeHistoryRef.current.ai.push(0);
        }
        
        // 履歴を保持（最大ENVELOPE_SAMPLES）
        if (envelopeHistoryRef.current.ai.length > ENVELOPE_SAMPLES) {
          envelopeHistoryRef.current.ai.shift(); // 左側（古い）を削除
        }
        
        // 表示用に配列をコピー（左から右に流れるように）
        const displayEnvelope = [...envelopeHistoryRef.current.ai];
        // 足りない分は0で埋める（左側）
        while (displayEnvelope.length < ENVELOPE_SAMPLES) {
          displayEnvelope.unshift(0);
        }
        
        // 平滑化を適用（より強い平滑化で安定させる）
        const smoothed = smoothEnvelope(displayEnvelope, 10);
        setAiEnvelope(smoothed);
        
        // デバッグ用（10フレームごと）
        if (frameCount % 10 === 0) {
          const maxAmp = Math.max(...smoothed);
          if (maxAmp > 0.01) {
            console.log("AI envelope:", { maxAmp, length: smoothed.length, nonZero: smoothed.filter(v => v > 0.01).length });
          }
        }
      }
      
      // Humanチャンネル
      if (humanAnalyser) {
        const amplitude = getCurrentAmplitude(humanAnalyser);
        
        if (amplitude > 0.01) {
          envelopeHistoryRef.current.human.push(amplitude);
        } else {
          envelopeHistoryRef.current.human.push(0);
        }
        
        if (envelopeHistoryRef.current.human.length > ENVELOPE_SAMPLES) {
          envelopeHistoryRef.current.human.shift();
        }
        
        const displayEnvelope = [...envelopeHistoryRef.current.human];
        while (displayEnvelope.length < ENVELOPE_SAMPLES) {
          displayEnvelope.unshift(0);
        }
        
        const smoothed = smoothEnvelope(displayEnvelope, 10);
        setHumanEnvelope(smoothed);
        
        // デバッグ用（10フレームごと）
        if (frameCount % 10 === 0) {
          const maxAmp = Math.max(...smoothed);
          if (maxAmp > 0.01) {
            console.log("Human envelope:", { maxAmp, length: smoothed.length, nonZero: smoothed.filter(v => v > 0.01).length });
          }
        }
      }
    }, updateInterval);

    return () => clearInterval(interval);
  }, [aiAnalyser, humanAnalyser]);

  // キャンバス描画
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      console.warn("WaveformVisualizer: canvas not found");
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      console.warn("WaveformVisualizer: canvas context not found");
      return;
    }

    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    console.log("WaveformVisualizer: canvas initialized", { width: WIDTH, height: HEIGHT });

    let drawCount = 0;
    const draw = () => {
      drawCount++;
      ctx.clearRect(0, 0, WIDTH, HEIGHT);

      // 背景
      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, WIDTH, HEIGHT);

      // タイトル
      ctx.fillStyle = "#8DBAE8";
      ctx.font = "bold 64px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("LLM-jp-Moshi", WIDTH / 2, 120);

      ctx.fillStyle = "#555555";
      ctx.font = "28px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillText(
        "A Japanese Full-duplex Spoken Dialogue System",
        WIDTH / 2,
        170
      );

      // 波形エリアのレイアウト
      const topMargin = 220;
      const bottomMargin = 80;
      const availableHeight = HEIGHT - topMargin - bottomMargin;
      const rowHeight = availableHeight / 2;
      const aiCenterY = topMargin + rowHeight / 2;
      const humanCenterY = topMargin + rowHeight + rowHeight / 2;
      const waveformWidth = WIDTH * 0.9;
      const leftX = WIDTH * 0.05;
      const rightX = leftX + waveformWidth;

      // 現在位置を 0〜1 に正規化
      const t = durationSec > 0 ? Math.min(Math.max(currentTimeSec / durationSec, 0), 1) : 0;
      const cursorX = leftX + waveformWidth * t;

      // エンベロープ描画関数
      const drawEnvelope = (
        env: number[],
        centerY: number,
        colorBase: string,
        alphaPast: number
      ) => {
        if (!env.length) return;
        const n = env.length;
        const stepX = waveformWidth / (n - 1);

        // リアルタイム表示：常に全波形を濃い色で表示
        // 振幅が0の場合は描画をスキップ（パフォーマンス向上）
        const hasData = env.some(v => v > 0.01);
        if (!hasData) return;
        
        ctx.beginPath();
        let pathStarted = false;
        
        for (let i = 0; i < n; i++) {
          const x = leftX + i * stepX;
          // 振幅を計算（0の場合は中心線に）
          const rawAmp = Math.max(env[i], 0);
          const amp = rawAmp * (rowHeight / 2) * 0.9; // 少し大きく表示
          const yTop = centerY - amp;

          if (!pathStarted) {
            ctx.moveTo(x, yTop);
            pathStarted = true;
          } else {
            ctx.lineTo(x, yTop);
          }
        }
        
        // 下側のパス
        for (let i = n - 1; i >= 0; i--) {
          const x = leftX + i * stepX;
          const rawAmp = Math.max(env[i], 0);
          const amp = rawAmp * (rowHeight / 2) * 0.9;
          const yBottom = centerY + amp;
          ctx.lineTo(x, yBottom);
        }
        
        ctx.closePath();
        // 常に濃い色で表示
        const alphaHex = Math.round(alphaPast * 255).toString(16).padStart(2, "0");
        ctx.fillStyle = `${colorBase}${alphaHex}`;
        ctx.fill();
        
        // アウトラインも描画（より見やすく）
        ctx.strokeStyle = colorBase;
        ctx.lineWidth = 1;
        ctx.stroke();

        // 中心線
        ctx.strokeStyle = "#DDDDDD";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(leftX, centerY);
        ctx.lineTo(rightX, centerY);
        ctx.stroke();
      };

      // AI ラベル
      ctx.fillStyle = "#A78DC9";
      ctx.font = "32px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("AI", leftX - 80, aiCenterY);

      // Human ラベル
      ctx.fillStyle = "#F5A0B4";
      ctx.fillText("Human", leftX - 120, humanCenterY);

      // AI 波形（紫）
      const aiHasData = aiEnvelope.some(v => v > 0.01);
      if (aiHasData) {
        drawEnvelope(aiEnvelope, aiCenterY, "#A78DC9", 0.85);
      }
      
      // AIテキストを波形の下に表示（カーソル位置に応じて、トークンずつ、色付け対応）
      if (aiText && aiText.length > 0 && tokenTimingsRef.current.length > 0) {
        // カーソル位置（t: 0〜1）までのトークンのみを表示
        const displayedTokenTimings: typeof tokenTimingsRef.current = [];
        
        for (const timing of tokenTimingsRef.current) {
          // トークンが生成された進行度がカーソル位置より前なら表示
          // timing.timeは0〜1の相対位置
          if (timing.time <= t) {
            displayedTokenTimings.push(timing);
          } else {
            // カーソル位置を超えたら終了
            break;
          }
        }
        
        // デバッグ用（100フレームごと）
        if (drawCount % 100 === 0 && displayedTokenTimings.length > 0) {
          const displayedText = displayedTokenTimings.map(t => t.token).join("");
          console.log("WaveformVisualizer: AI text", {
            totalTokens: tokenTimingsRef.current.length,
            displayedTokens: displayedTokenTimings.length,
            displayedText: displayedText.substring(0, 50),
            currentTimeSec,
            aiCenterY,
            rowHeight,
            humanCenterY,
          });
        }
        
        if (displayedTokenTimings.length > 0) {
          // テキストの位置：AI波形の下、Human波形の上
          const textY = aiCenterY + rowHeight / 2 + 20; // 波形の下に20pxのマージン
          
          // 色パレット（TextDisplayと同じ）
          const textDisplayColors = [
            "#d19bf7", "#d7acf6", "#debdf5", "#e4cef4",
            "#ebe0f3", "#eef2f0", "#c8ead9", "#a4e2c4",
            "#80d9af", "#5bd09a", "#38c886"
          ];
          
          function clamp_color(v: number) {
            return v <= 0
              ? 0
              : v >= textDisplayColors.length
                ? textDisplayColors.length - 1
                : Math.floor(v);
          }
          
          // 表示されるトークンの中で最新のインデックス
          const displayedCurrentIndex = displayedTokenTimings.length > 0 
            ? displayedTokenTimings[displayedTokenTimings.length - 1].index 
            : -1;
          
          ctx.textAlign = "left";
          ctx.textBaseline = "top";
          
          // テキストを右から左に流れるように表示
          // 最新のトークンがカーソル位置に表示され、古いトークンは左に流れる
          const lineHeight = 28;
          let y = textY;
          
          // カーソル位置から左に向かってテキストを描画
          // まず、表示するトークンの総幅を計算
          ctx.font = "normal 20px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
          let totalTextWidth = 0;
          const tokenWidths: number[] = [];
          
          for (const timing of displayedTokenTimings) {
            const tokenMetrics = ctx.measureText(timing.token);
            const tokenWidth = tokenMetrics.width;
            tokenWidths.push(tokenWidth);
            totalTextWidth += tokenWidth;
          }
          
          // カーソル位置から左に向かって描画（右から左へ）
          let currentX = cursorX;
          let currentLineWidth = 0;
          
          // トークンを逆順に処理（最新から古い順）
          for (let i = displayedTokenTimings.length - 1; i >= 0; i--) {
            const timing = displayedTokenTimings[i];
            const token = timing.token;
            const tokenWidth = tokenWidths[i];
            const isCurrentToken = timing.index === displayedCurrentIndex;
            const font = isCurrentToken 
              ? "bold 20px system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
              : "normal 20px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
            ctx.font = font;
            
            // 色を設定
            if (displayColor && timing.color !== undefined) {
              ctx.fillStyle = textDisplayColors[clamp_color(timing.color)];
            } else {
              ctx.fillStyle = "#333333";
            }
            
            // 左端を超えないようにする
            if (currentX - tokenWidth < leftX + 10) {
              // 左端を超える場合は、改行または描画を停止
              // 改行する場合
              y += lineHeight;
              currentX = cursorX;
              currentLineWidth = 0;
              
              // 画面外に出たら終了（Human波形の上まで）
              if (y > humanCenterY - rowHeight / 2 - 10) {
                break;
              }
            }
            
            // トークンを右から左に向かって描画
            currentX -= tokenWidth;
            ctx.fillText(token, currentX, y);
            currentLineWidth += tokenWidth;
          }
        }
      } else {
        // デバッグ用：テキストがない場合
        if (drawCount % 200 === 0) {
          console.log("WaveformVisualizer: No AI text", {
            aiText,
            aiTextLength: aiText?.length,
          });
        }
      }
      
      // Human 波形（ピンク）
      const humanHasData = humanEnvelope.some(v => v > 0.01);
      if (humanHasData) {
        drawEnvelope(humanEnvelope, humanCenterY, "#F5A0B4", 0.85);
      }

      // デバッグ用（100フレームごと）
      if (drawCount % 100 === 0) {
        console.log("WaveformVisualizer: draw called", {
          drawCount,
          aiEnvelopeLength: aiEnvelope.length,
          aiMaxAmp: Math.max(...aiEnvelope, 0),
          humanEnvelopeLength: humanEnvelope.length,
          humanMaxAmp: Math.max(...humanEnvelope, 0),
          aiHasData,
          humanHasData,
          aiNonZero: aiEnvelope.filter(v => v > 0.01).length,
          humanNonZero: humanEnvelope.filter(v => v > 0.01).length,
        });
      }
      
      // 初回描画時にもログ出力
      if (drawCount === 1) {
        console.log("WaveformVisualizer: first draw", {
          aiEnvelope: aiEnvelope.slice(0, 10),
          humanEnvelope: humanEnvelope.slice(0, 10),
          aiHasData,
          humanHasData,
        });
      }

      // カーソル（緑）
      ctx.strokeStyle = "#7FD9B3";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cursorX, topMargin - 10);
      ctx.lineTo(cursorX, HEIGHT - bottomMargin + 10);
      ctx.stroke();
    };

    // アニメーション更新
    const interval = setInterval(draw, 1000 / FPS);
    draw(); // 初回描画
    return () => clearInterval(interval);
  }, [aiEnvelope, humanEnvelope, durationSec, currentTimeSec, aiText, aiTextColor, displayColor]);

  return (
    <div
      style={{
        width: "100%",
        aspectRatio: "16 / 9",
        backgroundColor: "#FFFFFF",
        position: "relative",
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          height: "100%",
          display: "block",
          position: "absolute",
          top: 0,
          left: 0,
        }}
      />
    </div>
  );
};
