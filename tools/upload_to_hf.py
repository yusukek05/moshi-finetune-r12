#!/usr/bin/env python3
"""
Hugging Face Hubにチェックポイントをアップロードするスクリプト
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import tempfile
import shutil


def upload_checkpoint_to_hf(
    checkpoint_dir: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload checkpoint",
    source_repo: str = None,
):
    """
    チェックポイントをHugging Face Hubにアップロード

    Args:
        checkpoint_dir: チェックポイントディレクトリのパス
        repo_id: Hugging Face HubのリポジトリID (例: "username/model-name")
        private: プライベートリポジトリとして作成するか
        commit_message: コミットメッセージ
        source_repo: トークナイザーファイルのソースリポジトリID (例: "nu-dialogue/j-moshi")
    """
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise ValueError(f"チェックポイントディレクトリが存在しません: {checkpoint_dir}")

    # 必要なファイルを確認
    model_file = checkpoint_path / "model.safetensors"
    config_file = checkpoint_path / "moshi_lm_kwargs.json"
    
    # トークナイザーファイル（オプション）
    speech_tokenizer_file = checkpoint_path / "tokenizer-e351c8d8-checkpoint125.safetensors"
    text_tokenizer_file = checkpoint_path / "tokenizer_spm_32k_3.model"

    if not model_file.exists():
        raise ValueError(f"model.safetensorsが見つかりません: {model_file}")
    if not config_file.exists():
        raise ValueError(f"moshi_lm_kwargs.jsonが見つかりません: {config_file}")

    # 一時ディレクトリの作成（ダウンロードしたファイル用）
    temp_dir = None

    # トークナイザーファイルが存在しない場合、ソースリポジトリからダウンロード
    if source_repo:
        if not speech_tokenizer_file.exists():
            print(f"tokenizer-e351c8d8-checkpoint125.safetensorsを '{source_repo}' からダウンロード中...")
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            try:
                downloaded_path = hf_hub_download(
                    repo_id=source_repo,
                    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
                    local_dir=temp_dir,
                )
                speech_tokenizer_file = Path(downloaded_path)
                print(f"✓ ダウンロード完了: {speech_tokenizer_file}")
            except Exception as e:
                print(f"⚠ ダウンロードに失敗しました: {e}")
                speech_tokenizer_file = None

        if not text_tokenizer_file.exists():
            print(f"tokenizer_spm_32k_3.modelを '{source_repo}' からダウンロード中...")
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            try:
                downloaded_path = hf_hub_download(
                    repo_id=source_repo,
                    filename="tokenizer_spm_32k_3.model",
                    local_dir=temp_dir,
                )
                text_tokenizer_file = Path(downloaded_path)
                print(f"✓ ダウンロード完了: {text_tokenizer_file}")
            except Exception as e:
                print(f"⚠ ダウンロードに失敗しました: {e}")
                text_tokenizer_file = None

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

    # トークナイザーファイルのアップロード（存在する場合）
    if speech_tokenizer_file and speech_tokenizer_file.exists():
        print(f"tokenizer-e351c8d8-checkpoint125.safetensorsをアップロード中...")
        upload_file(
            path_or_fileobj=str(speech_tokenizer_file),
            path_in_repo="tokenizer-e351c8d8-checkpoint125.safetensors",
            repo_id=repo_id,
            commit_message=commit_message,
        )
        print("✓ tokenizer-e351c8d8-checkpoint125.safetensorsのアップロードが完了しました")
    else:
        print("⚠ tokenizer-e351c8d8-checkpoint125.safetensorsが見つかりません（スキップします）")

    if text_tokenizer_file and text_tokenizer_file.exists():
        print(f"tokenizer_spm_32k_3.modelをアップロード中...")
        upload_file(
            path_or_fileobj=str(text_tokenizer_file),
            path_in_repo="tokenizer_spm_32k_3.model",
            repo_id=repo_id,
            commit_message=commit_message,
        )
        print("✓ tokenizer_spm_32k_3.modelのアップロードが完了しました")
    else:
        print("⚠ tokenizer_spm_32k_3.modelが見つかりません（スキップします）")

    # README.mdの作成（オプション）
    readme_content = f"""---
license: cc-by-nc-4.0
datasets:
- sarulab-speech/J-CHAT
language:
- ja
base_model:
- kyutai/moshiko-pytorch-bf16
library_name: moshi
---

j-moshi-v1.0

このモデルは、Moshiモデルをファインチューニングしたものです。

### Installation
Python 3.10以上が必要です．

```bash
pip install moshi<=0.2.2
```

### Usage
`moshi.server`を実行することで，対話用のweb UIを起動できます．`--hf-repo`オプションでJ-Moshiの 🤗HuggingFace Hubリポジトリ（[llm-jp/j-moshi-v1](https://huggingface.co/llm-jp/j-moshi-v1)）を指定してください．

```bash
python -m moshi.server --hf-repo llm-jp/j-moshi-v1
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

    # 一時ディレクトリのクリーンアップ
    if temp_dir and Path(temp_dir).exists():
        try:
            shutil.rmtree(temp_dir)
            print(f"✓ 一時ファイルをクリーンアップしました")
        except Exception as e:
            print(f"⚠ 一時ディレクトリのクリーンアップに失敗しました: {e}")

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
    parser.add_argument(
        "--source-repo",
        type=str,
        default=None,
        help="トークナイザーファイルのソースリポジトリID (例: nu-dialogue/j-moshi)",
    )

    args = parser.parse_args()

    upload_checkpoint_to_hf(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        source_repo=args.source_repo,
    )


if __name__ == "__main__":
    main()

