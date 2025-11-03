# init_moshi_llmjp_ft.py
import argparse
import os
from copy import deepcopy

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models import MoshiLlamaForFinetuning, extend_moshi_modules_for_user_stream
from models.modeling_moshi_llama import MoshiLlama  # あなたの実装


def main(args):
    device = "cpu"
    dtype = getattr(torch, args.model_dtype)

    # 1) HF から LLM-jp を取得
    cfg = AutoConfig.from_pretrained(args.hf_repo, trust_remote_code=False)
    tok = AutoTokenizer.from_pretrained(
        args.hf_repo, use_fast=True, trust_remote_code=False
    )
    print("HF vocab_size:", cfg.vocab_size, "pad_token_id:", tok.pad_token_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_repo, torch_dtype=dtype, device_map=None, trust_remote_code=False
    )

    delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]

    # 2) MoshiLlama を “HF Config に合わせた値” で構築
    moshi_llama = MoshiLlama(
        llama_name_or_path=args.hf_repo,  # あなたの __init__ ではここから LlamaConfig を再取得
        # ★ 下の3つは assert を通すために必須（HF Config から渡す）
        text_card=cfg.vocab_size,
        dim=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        # ★ ここから下は用途に合わせて。デフォルトのままでもOK
        delays=delays,
        n_q=args.n_q,
        dep_q=args.dep_q,
        depformer_dim=args.depformer_dim,
        depformer_dim_feedforward=(
            args.depformer_dim_ff
            if args.depformer_dim_ff is not None
            else int(args.hidden_scale * args.depformer_dim)
        ),
        depformer_pos_emb=args.depformer_pos_emb,
        depformer_multi_linear=args.depformer_multi_linear,
        depformer_weights_per_step=args.depformer_weights_per_step,
        existing_text_padding_id=None,  # tokenizer に padding が無いなら None（+1 行ぶん確保）
        end_of_text_padding_id=None,
        hidden_scale=args.hidden_scale,
        norm=args.norm,
        norm_emb=args.norm_emb,
        bias_proj=args.bias_proj,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)

    # 3) HF → MoshiLlama.transformer へ重み移植（lm_head/embedding は除外）
    sd = hf_model.state_dict()
    mapped = {}
    for k, v in sd.items():
        if k.startswith("lm_head."):
            continue
        if k.startswith("model."):
            k = k[len("model.") :]  # 'model.' を剥がす
        mapped[k] = v
    missing, unexpected = moshi_llama.transformer.load_state_dict(mapped, strict=False)
    print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

    # 4) 可能なら text_emb に HF の embed_tokens をコピー（padding 行(+1)に注意）
    #    MoshiLlama は text_emb を (vocab_size + (padding有なら+1)) で作成しています。
    hf_emb = sd.get("model.embed_tokens.weight", None)
    if hf_emb is not None and moshi_llama.text_emb.weight.shape[0] >= hf_emb.shape[0]:
        with torch.no_grad():
            moshi_llama.text_emb.weight[: hf_emb.shape[0]].copy_(hf_emb.to(dtype))
        print("Copied HF embed_tokens -> moshi_llama.text_emb (first vocab rows).")
    else:
        print("Skip copying embed_tokens (not found or shape mismatch).")

    # 5) （任意）depformer を user-stream 対応に拡張
    if args.extend_modules_for_user_stream:
        print("Extending depformer for user stream ...")
        moshi_llama = extend_moshi_modules_for_user_stream(moshi_llama)
        # 学習側の設定とも一致させる（例: 8+8=16）
        # ここでは kwargs を後で保存するため、メタ情報として持たせたい場合は別途 dict に記録して保存してください。

    # 6) MoshiLlamaForFinetuning にラップ（ZeRO-3 互換の forward パッチを depformer に適用）
    moshi_llama_ft = MoshiLlamaForFinetuning.from_original_moshi_llama(
        moshi_llama=moshi_llama,
        moshi_llama_kwargs=dict(  # ★ “後で to_original” する用に保存されます
            llama_name_or_path=args.hf_repo,
            n_q=args.n_q,
            dep_q=args.dep_q,
            card=args.card,
            text_card=cfg.vocab_size,
            dim=cfg.hidden_size,
            num_heads=cfg.num_heads,
            hidden_scale=args.hidden_scale,
            norm=args.norm,
            norm_emb=args.norm_emb,
            bias_proj=args.bias_proj,
            depformer_dim=args.depformer_dim,
            depformer_dim_feedforward=(
                args.depformer_dim_ff
                if args.depformer_dim_ff is not None
                else int(args.hidden_scale * args.depformer_dim)
            ),
            depformer_multi_linear=args.depformer_multi_linear,
            depformer_weights_per_step=args.depformer_weights_per_step,
            depformer_pos_emb=args.depformer_pos_emb,
            existing_text_padding_id=None,
            end_of_text_padding_id=None,
            context=None,
            device=device,
            dtype=str(dtype).split(".")[-1],
        ),
    ).to(device=device, dtype=dtype)

    # 7) 保存
    os.makedirs(args.save_dir, exist_ok=True)
    moshi_llama_ft.save_pretrained(args.save_dir)
    tok.save_pretrained(args.save_dir)  # tokenizer も一緒に置くと後工程が楽
    print(f"Saved to: {args.save_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--hf_repo", type=str, default="llm-jp/llm-jp-3-7.2b")
    p.add_argument(
        "--model_dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )

    # Moshi 側（あなたの実装に合わせて適宜変更）
    p.add_argument("--n_q", type=int, default=8)
    p.add_argument("--dep_q", type=int, default=8)
    p.add_argument("--card", type=int, default=1024)

    p.add_argument("--depformer_dim", type=int, default=256)
    p.add_argument("--depformer_dim_ff", type=int, default=None)
    p.add_argument("--depformer_pos_emb", type=str, default="sin")
    p.add_argument("--depformer_multi_linear", action="store_true")
    p.add_argument("--depformer_weights_per_step", action="store_true")

    p.add_argument("--hidden_scale", type=int, default=4)
    p.add_argument("--norm", type=str, default="layer_norm")
    p.add_argument("--norm_emb", action="store_true")
    p.add_argument("--bias_proj", action="store_true")

    p.add_argument("--extend_modules_for_user_stream", action="store_true")
    args = p.parse_args()
    main(args)
