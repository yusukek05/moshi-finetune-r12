#!/bin/bash
# クライアントUIをビルドするスクリプト

cd /home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/client

# Node.jsが利用可能か確認
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed or not in PATH"
    echo "Please install Node.js (version 18 or higher recommended)"
    exit 1
fi

# npm install（初回のみ、または依存関係が変更された場合）
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# ビルド実行
echo "Building client..."
npm run build

if [ $? -eq 0 ]; then
    echo "Build completed successfully!"
    echo "Output directory: client/dist"
else
    echo "Build failed!"
    exit 1
fi
