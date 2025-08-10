import json
import os
import re
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from moshi.models import LMModel
from moshi.modules.gating import gating_forward_kernel
from moshi.modules.transformer import (
    StreamingTransformer,
    create_sin_embedding,
    multi_linear,
)
from safetensors.torch import load_model, save_file

# 追加: LLM-jp をロード
from transformers import AutoModelForCausalLM, AutoTokenizer


def expose_linear_weights_for_zero3_depformer_only(
    moshi_lm: LMModel,
) -> None:
    """
    DeepSpeed ZeRO-3 互換化: depformer 側のみ子 Linear を露出して元 Linear を削除。
    tempformer を LLM 置換する場合はこちらを使う。
    Target:
      - depformer.layers[*].gating[*].linear_in/out
      - depformer.layers[*].self_attn.out_proj
    """
    if isinstance(moshi_lm.depformer, StreamingTransformer):
        for layer in moshi_lm.depformer.layers:
            for gating in layer.gating:
                gating.linear_in_weight = gating.linear_in.weight
                gating.linear_out_weight = gating.linear_out.weight
                del gating.linear_in, gating.linear_out
            layer.self_attn.out_proj_weight = layer.self_attn.out_proj.weight
            del layer.self_attn.out_proj


def expose_linear_weights_for_zero3_both(moshi_lm: LMModel) -> None:
    """
    従来どおり transformer/depformer の両方に対して露出を行う。
    """
    if isinstance(moshi_lm.transformer, StreamingTransformer):
        for layer in moshi_lm.transformer.layers:
            layer.gating.linear_in_weight = layer.gating.linear_in.weight
            layer.gating.linear_out_weight = layer.gating.linear_out.weight
            del layer.gating.linear_in, layer.gating.linear_out
    expose_linear_weights_for_zero3_depformer_only(moshi_lm)


def activation_gating_forward(self, x: torch.Tensor):
    """Exposed weights を使う版"""
    return gating_forward_kernel(
        self.linear_in_weight,
        self.linear_out_weight,
        self.activation,
        x,
    )


def mha_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    """Exposed weights を使う版"""
    state = self._streaming_state
    T = query.shape[1]

    if state is None:
        offset = torch.zeros(1, device=query.device, dtype=torch.long)
        offset_cpu = 0
    else:
        assert self.causal, "Streaming only available for causal"
        offset = state.offset
        offset_cpu = state.offset_cpu

    if self.weights_per_step:
        projected = multi_linear(
            self.weights_per_step, self.in_proj_weight, query, offset_cpu
        )
    else:
        projected = nn.functional.linear(query, self.in_proj_weight)
    q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)

    if self.rope:
        q, k = self.rope(q, k, offset, time_before_heads=False)

    k, v, pos_k = self._complete_kv(k, v)
    if self.causal:
        pos_k = pos_k.view(1, -1)
        pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)
        if self.context is not None:
            attn_bias = attn_bias & (delta < self.context)
    else:
        attn_bias = None
    x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)

    x = rearrange(x, "b h t d -> b t (h d)")
    if self.weights_per_step:
        x = multi_linear(self.weights_per_step, self.out_proj_weight, x, offset_cpu)
    else:
        x = self.out_proj(x)
    if state is not None:
        state.offset.add_(T)
        state.offset_cpu += T
    return x


def transformer_forward(self, x: torch.Tensor, *args, **kwargs):
    """
    StreamingTransformer の通常 forward（activation checkpoint 切替付き）
    """
    B, T, C = x.shape

    state = self._streaming_state
    if state is None:
        offset = torch.zeros(1, dtype=torch.long, device=x.device)
    else:
        offset = state.offset

    if self.positional_embedding in {"sin", "sin_rope"}:
        positions = torch.arange(T, device=x.device).view(1, -1, 1)
        positions = positions + offset.view(-1, 1, 1)
        pos_emb = create_sin_embedding(
            positions, C, max_period=self.max_period, dtype=x.dtype
        )
        x = x + self.positional_scale * pos_emb

    for layer in self.layers:
        if self.activation_checkpointing:
            x = self.checkpointing_func(layer, x)
        else:
            x = layer(x)

    if state is not None:
        state.offset.add_(T)
    return x


def restore_linear_weights_from_exposed_state_dict(
    moshi_lm_for_ft_state_dict: OrderedDict,
) -> OrderedDict:
    """
    Exposed state dict → 元の Linear 名称へ戻す。
    """
    gating_linear_in_pattern = re.compile(
        r"transformer\.layers\.\d+\.gating\.linear_in_weight"
    )
    gating_linear_out_pattern = re.compile(
        r"transformer\.layers\.\d+\.gating\.linear_out_weight"
    )
    depformer_gating_linear_in_pattern = re.compile(
        r"depformer\.layers\.\d+\.gating\.\d+\.linear_in_weight"
    )
    depformer_gating_linear_out_pattern = re.compile(
        r"depformer\.layers\.\d+\.gating\.\d+\.linear_out_weight"
    )
    depformer_self_attn_out_proj_pattern = re.compile(
        r"depformer\.layers\.\d+\.self_attn\.out_proj_weight"
    )

    new_state_dict = OrderedDict()
    for key in moshi_lm_for_ft_state_dict.keys():
        if gating_linear_in_pattern.match(key):
            new_key = key.replace("linear_in_weight", "linear_in.weight")
        elif gating_linear_out_pattern.match(key):
            new_key = key.replace("linear_out_weight", "linear_out.weight")
        elif depformer_gating_linear_in_pattern.match(key):
            new_key = key.replace("linear_in_weight", "linear_in.weight")
        elif depformer_gating_linear_out_pattern.match(key):
            new_key = key.replace("linear_out_weight", "linear_out.weight")
        elif depformer_self_attn_out_proj_pattern.match(key):
            new_key = key.replace("out_proj_weight", "out_proj.weight")
        else:
            new_key = key
        if new_key != key:
            print(f"{key} -> {new_key}")
        new_state_dict[new_key] = moshi_lm_for_ft_state_dict[key]

    return new_state_dict


# ─────────────────────────────────────────────────────────────────────────────
# LLM を tempformer として使うための薄いラッパ
# ─────────────────────────────────────────────────────────────────────────────
class LLMTempformerWrapper(nn.Module):
    """
    HF の CausalLM を `inputs_embeds` で駆動し、last_hidden_state を返すだけのラッパ。
    Moshi の `transformer(x, attention_mask=...)` 互換のシグネチャにする。
    """

    def __init__(self, llm: nn.Module):
        super().__init__()
        # 注意: ここで llm を submodule として登録（学習で微調整したい場合に備える）
        self.llm = llm

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, *_, **__
    ):
        out = self.llm(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        return out.last_hidden_state


class MoshiLLMJP3ForFinetuning(LMModel):
    """
    Moshi language model for finetuning (tempformer=StreamingTransformer or LLM).
    """

    def __init__(self, *args, **kwargs):
        # 親に渡さないメタ情報
        self.llm_tempformer_cfg = kwargs.pop("llm_tempformer", None)
        self._defer_llm_init = kwargs.pop("defer_llm_init", False)
        super().__init__(*args, **kwargs)

        # tempformer: LLM を使わない従来ルート
        if self.llm_tempformer_cfg is None:
            # 変換: transformer/depformer 両方
            expose_linear_weights_for_zero3_both(self)

            # forward パッチ（transformer & depformer）
            for layer in self.transformer.layers:
                layer.gating.forward = activation_gating_forward.__get__(layer.gating)
            for layer in self.depformer.layers:
                for gating in layer.gating:
                    gating.forward = activation_gating_forward.__get__(gating)
            for layer in self.depformer.layers:
                layer.self_attn.forward = mha_forward.__get__(layer.self_attn)

            # Activation checkpointing switch
            self.transformer.activation_checkpointing = False
            self.transformer.forward = transformer_forward.__get__(self.transformer)
            self.depformer.activation_checkpointing = False
            self.depformer.forward = transformer_forward.__get__(self.depformer)
            return  # ── 従来完了 ──

        # ここから LLM tempformer ルート
        # depformer 側のみ ZeRO-3 互換化
        expose_linear_weights_for_zero3_depformer_only(self)
        for layer in self.depformer.layers:
            for gating in layer.gating:
                gating.forward = activation_gating_forward.__get__(gating)
        for layer in self.depformer.layers:
            layer.self_attn.forward = mha_forward.__get__(layer.self_attn)
        self.depformer.activation_checkpointing = False
        self.depformer.forward = transformer_forward.__get__(self.depformer)

        # LLM の初期化は、from_original_moshi_lm() 直後の strict=True ロードを通すために
        # 必要に応じて遅延させる。
        if not self._defer_llm_init:
            self._init_llm_tempformer()

    # LLM を組み込んで tempformer を置換
    def _init_llm_tempformer(self):
        if getattr(self, "_llm_initialized", False):
            return  # 二重初期化防止

        cfg = self.llm_tempformer_cfg
        assert cfg is not None, "llm_tempformer_cfg is None"
        repo = cfg.get("repo")
        revision = cfg.get("revision", None)

        # CPU でも安全なように fp32 でロード（後で Accelerator が移動/変換）
        llm = AutoModelForCausalLM.from_pretrained(
            repo,
            revision=revision,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )
        # Tokenizer 情報は必要なら参照するが、ここでは保持不要
        # tok = AutoTokenizer.from_pretrained(repo, revision=revision)

        # 1) Moshi の transformer を LLM ラッパに置換
        self.transformer = LLMTempformerWrapper(llm)

        # 2) テキスト埋め込み/出力層を LLM に差し替え
        self.text_emb = llm.get_input_embeddings()
        self.text_linear = llm.lm_head

        # 3) tempformer 用の audio 域を LLM hidden に射影するため、
        #    emb[acb] を Embedding -> Linear の逐次に置換
        llm_hidden = llm.config.hidden_size
        for i in range(self.num_audio_codebooks):
            old_emb: nn.Embedding = self.emb[i]
            in_dim = old_emb.embedding_dim
            proj = nn.Linear(in_dim, llm_hidden, bias=False)
            # 既存の埋め込み重みは保持し、線形は新規初期化
            seq = nn.Sequential(old_emb, proj)
            self.emb[i] = seq  # 以後 tempformer_forward の audio_emb は LLM 次元になる

        self._llm_initialized = True

    def enable_activation_checkpointing(self, checkpointing_func):
        # LLM tempformer でも depformer 側には有効化する
        if self.llm_tempformer_cfg is None:
            self.transformer.activation_checkpointing = True
            self.transformer.checkpointing_func = checkpointing_func
        self.depformer.activation_checkpointing = True
        self.depformer.checkpointing_func = checkpointing_func

    def disable_activation_checkpointing(self):
        if self.llm_tempformer_cfg is None:
            self.transformer.activation_checkpointing = False
        self.depformer.activation_checkpointing = False

    @classmethod
    def from_original_moshi_lm(
        cls,
        moshi_lm: LMModel,
        moshi_lm_kwargs: dict,
    ) -> "MoshiLLMJP3ForFinetuning":
        """
        Initialize `MoshiLLMJP3ForFinetuning` from the original `LMModel`.
        """
        # 旧 Moshi を ZeRO-3 互換化して state_dict 抜き出し
        # （transformer 側は触らない方が無難なのでここでは触らない）
        state_dict = moshi_lm.state_dict()
        device = next(moshi_lm.parameters()).device
        dtype = next(moshi_lm.parameters()).dtype
        del moshi_lm

        # LLM の登録で strict=True を壊さないよう、最初は遅延初期化にする
        init_kwargs = dict(moshi_lm_kwargs)
        init_kwargs["defer_llm_init"] = True

        moshi_lm = cls(device=device, dtype=dtype, **init_kwargs).to(
            device=device, dtype=dtype
        )
        # 旧 Moshi の重みを strict にロード（LLM は未登録のため問題なし）
        missing, unexpected = moshi_lm.load_state_dict(state_dict, strict=False)
        # ↑ torch の戻り値は (missing, unexpected) だが strict=True なら通常例外になる想定

        # strict ロード後に LLM を組み込む
        if moshi_lm.llm_tempformer_cfg is not None:
            moshi_lm._init_llm_tempformer()

        # kwargs を保持（JSON 保存用）
        moshi_lm.moshi_lm_kwargs = moshi_lm_kwargs
        return moshi_lm

    def to_original_moshi_lm(self) -> LMModel:
        """
        Convert the model to the original `LMModel`.
        """
        state_dict = self.state_dict()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Exposed -> 元 Linear 名称へ戻す
        state_dict = restore_linear_weights_from_exposed_state_dict(state_dict)

        # 元モデル初期化（未知キーは渡さない）
        init_kwargs = dict(self.moshi_lm_kwargs)
        init_kwargs.pop("llm_tempformer", None)

        moshi_lm = LMModel(device=device, dtype=dtype, **init_kwargs).to(
            device=device, dtype=dtype
        )
        moshi_lm.load_state_dict(state_dict, strict=True)
        return moshi_lm

    def save_pretrained(self, save_dir: str):
        """
        Save the model to the given directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        # 注意: 現状は Moshi 側重み＋（登録済みなら）LLM 側も state_dict に含まれます。
        # ファイルサイズが大きい場合は、LLM を state_dict から除外する拡張も検討してください。
        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))
        with open(os.path.join(save_dir, "moshi_lm_kwargs.json"), "w") as f:
            json.dump(self.moshi_lm_kwargs, f, indent=4)

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MoshiLLMJP3ForFinetuning":
        """
        Load the model from the given directory.
        """
        with open(os.path.join(save_dir, "moshi_lm_kwargs.json")) as f:
            moshi_lm_kwargs = json.load(f)

        # LLM を含めて初期化（ここでは遅延させず OK。safetensors に LLM が無くても問題なし）
        moshi_lm = cls(device=device, dtype=dtype, **moshi_lm_kwargs).to(
            device=device, dtype=dtype
        )
        # Moshi 側の重みをロード（存在するキーのみ読み込まれる）
        load_model(moshi_lm, os.path.join(save_dir, "model.safetensors"))

        moshi_lm.moshi_lm_kwargs = moshi_lm_kwargs
        return moshi_lm
