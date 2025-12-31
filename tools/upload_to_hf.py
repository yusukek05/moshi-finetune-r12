#!/usr/bin/env python3
"""
Hugging Face Hubにチェックポイントをアップロードするスクリプト
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file
from huggingface_hub.utils import HfHubHTTPError


def upload_checkpoint_to_hf(
    checkpoint_dir: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload checkpoint",
):
    """
    チェックポイントをHugging Face Hubにアップロード

    Args:
        checkpoint_dir: チェックポイントディレクトリのパス
        repo_id: Hugging Face HubのリポジトリID (例: "username/model-name")
        private: プライベートリポジトリとして作成するか
        commit_message: コミットメッセージ
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise ValueError(f"チェックポイントディレクトリが存在しません: {checkpoint_dir}")

    # 必要なファイルを確認
    model_file = checkpoint_path / "model.safetensors"
    config_file = checkpoint_path / "moshi_lm_kwargs.json"

    if not model_file.exists():
        raise ValueError(f"model.safetensorsが見つかりません: {model_file}")
    if not config_file.exists():
        raise ValueError(f"moshi_lm_kwargs.jsonが見つかりません: {config_file}")

    # Hugging Face APIの初期化
    api = HfApi()

    # リポジトリの作成（存在しない場合）
    try:
        create_repo(repo_id=repo_id, private=private, exist_ok=True)
        print(f"リポジトリ '{repo_id}' を作成または確認しました")
    except HfHubHTTPError as e:
        if "already exists" in str(e).lower():
            print(f"リポジトリ '{repo_id}' は既に存在します")
        else:
            raise

    # ファイルのアップロード
    print(f"model.safetensorsをアップロード中...")
    upload_file(
        path_or_fileobj=str(model_file),
        path_in_repo="model.safetensors",
        repo_id=repo_id,
        commit_message=commit_message,
    )
    print("✓ model.safetensorsのアップロードが完了しました")

    print(f"moshi_lm_kwargs.jsonをアップロード中...")
    upload_file(
        path_or_fileobj=str(config_file),
        path_in_repo="moshi_lm_kwargs.json",
        repo_id=repo_id,
        commit_message=commit_message,
    )
    print("✓ moshi_lm_kwargs.jsonのアップロードが完了しました")

    # README.mdの作成（オプション）
    readme_content = f"""---
license: apache-2.0
base_model: kyutai/moshiko-pytorch-bf16
tags:
- moshi
- audio
- dialogue
- japanese
---

# {repo_id.split('/')[-1]}

このモデルは、Moshiモデルをファインチューニングしたものです。

## 使用方法

```python
from moshi.models import MimiModel, loaders
from huggingface_hub import hf_hub_download

# モデルの読み込み
model_path = hf_hub_download(repo_id="{repo_id}", filename="model.safetensors")
config_path = hf_hub_download(repo_id="{repo_id}", filename="moshi_lm_kwargs.json")

with open(config_path) as f:
    config = json.load(f)

model = loaders.load_moshi_lm(model_path, **config)
```
"""
    readme_path = checkpoint_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"README.mdをアップロード中...")
    upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message=commit_message,
    )
    print("✓ README.mdのアップロードが完了しました")

    print(f"\n✓ アップロードが完了しました！")
    print(f"リポジトリURL: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="チェックポイントをHugging Face Hubにアップロード"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="チェックポイントディレクトリのパス",
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="Hugging Face HubのリポジトリID (例: username/model-name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="プライベートリポジトリとして作成",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload checkpoint",
        help="コミットメッセージ",
    )

    args = parser.parse_args()

    upload_checkpoint_to_hf(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()

