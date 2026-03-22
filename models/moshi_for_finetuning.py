import json
import os
import re
import functools
from glob import glob
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from moshi.models import LMModel
from moshi.modules.gating import (
    gating_forward_kernel,
)
from moshi.modules.transformer import (
    StreamingTransformer,
    create_sin_embedding,
    multi_linear,
)
from safetensors.torch import load_model, save_file

from models.modeling_moshi_llama import MoshiLlama


def expose_linear_weights_for_zero3(
    moshi_lm: LMModel,
) -> None:
    """Exposes linear layer weights at their parent modules for DeepSpeed Zero-3 compatibility.

    In DeepSpeed Zero-3, weights of child modules cannot be accessed directly, so we need to
    restructure the model by exposing weights from linear layers at their
    parent modules while removing the original linear modules to maintain compatibility.

    Target modules:
    - `transformer.layers[*].gating.linear_in`
    - `transformer.layers[*].gating.linear_out`
    - `depformer.layers[*].gating[*].linear_in`
    - `depformer.layers[*].gating[*].linear_out`
    - `depformer.layers[*].self_attn.out_proj`
    """
    if isinstance(moshi_lm.transformer, StreamingTransformer):
        for layer in moshi_lm.transformer.layers:
            layer.gating.linear_in_weight = layer.gating.linear_in.weight
            layer.gating.linear_out_weight = layer.gating.linear_out.weight
            del layer.gating.linear_in, layer.gating.linear_out
    if isinstance(moshi_lm.depformer, StreamingTransformer):
        for layer in moshi_lm.depformer.layers:
            for gating in layer.gating:
                gating.linear_in_weight = gating.linear_in.weight
                gating.linear_out_weight = gating.linear_out.weight
                del gating.linear_in, gating.linear_out
            layer.self_attn.out_proj_weight = layer.self_attn.out_proj.weight
            del layer.self_attn.out_proj


def activation_gating_forward(self, x: torch.Tensor):
    """
    Most of the code in this function is copied from `moshi.modules.gating.ActivationGating.forward`
    with the exception of accessing own linear weights.
    """
    return gating_forward_kernel(
        # use the exposed linear weights
        self.linear_in_weight,
        self.linear_out_weight,
        self.activation,
        x,
    )


def mha_forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    """
    Most of the code in this function is copied from `moshi.modules.transformer.StreamingMultiheadAttention.forward`
    with the exception of accessing own linear weights.
    """
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
        projected = multi_linear(self.weights_per_step, self.in_proj_weight, query, offset_cpu)
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
        # Use the exposed linear weights (out_proj_weight)
        x = multi_linear(self.weights_per_step, self.out_proj_weight, x, offset_cpu)
    else:
        x = self.out_proj(x)
    if state is not None:
        state.offset.add_(T)
        state.offset_cpu += T
    return x


def transformer_forward(self, x: torch.Tensor, *args, **kwargs):
    """
    Most of the code in this function is copied from `moshi.modules.transformer.StreamingTransformer.forward`
    with the exception of activation checkpointing.
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
        pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
        x = x + self.positional_scale * pos_emb

    for layer in self.layers:
        if self.activation_checkpointing:
            x = self.checkpointing_func(layer, x)
        else:
            x = layer(x)  # , *args, **kwargs)

    if state is not None:
        state.offset.add_(T)
    return x


def restore_linear_weights_from_exposed_state_dict(
    moshi_lm_for_ft_state_dict: OrderedDict,
) -> OrderedDict:
    """
    Restore linear layer weights from the exposed state dict for DeepSpeed Zero-3 compatibility.

    Target parameters:
    - `transformer.layers[*].gating.linear_in_weight`
    - `transformer.layers[*].gating.linear_out_weight`
    - `depformer.layers[*].gating[*].linear_in_weight`
    - `depformer.layers[*].gating[*].linear_out_weight`
    - `depformer.layers[*].self_attn.out_proj_weight`
    """

    gating_linear_in_pattern = re.compile(r"transformer\.layers\.\d+\.gating\.linear_in_weight")
    gating_linear_out_pattern = re.compile(r"transformer\.layers\.\d+\.gating\.linear_out_weight")
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


class MoshiForFinetuning(LMModel):
    """
    Moshi language model for finetuning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # DeepSpeed Zero-3 compatibility
        ## 1. Expose linear layer weights
        expose_linear_weights_for_zero3(self)
        ## 2. Apply patches for forward functions
        for layer in self.transformer.layers:
            layer.gating.forward = activation_gating_forward.__get__(layer.gating)
        for layer in self.depformer.layers:
            for gating in layer.gating:
                gating.forward = activation_gating_forward.__get__(gating)
        for layer in self.depformer.layers:
            layer.self_attn.forward = mha_forward.__get__(layer.self_attn)

        # Implement activation checkpointing
        self.transformer.activation_checkpointing = False
        self.transformer.forward = transformer_forward.__get__(self.transformer)
        self.depformer.activation_checkpointing = False
        self.depformer.forward = transformer_forward.__get__(self.depformer)

    def enable_activation_checkpointing(self, checkpointing_func):
        self.transformer.activation_checkpointing = True
        self.transformer.checkpointing_func = checkpointing_func
        self.depformer.activation_checkpointing = True
        self.depformer.checkpointing_func = checkpointing_func

    def disable_activation_checkpointing(self):
        self.transformer.activation_checkpointing = False
        self.depformer.activation_checkpointing = False

    @classmethod
    def from_original_moshi_lm(
        cls,
        moshi_lm: LMModel,
        moshi_lm_kwargs: dict,
    ) -> "MoshiForFinetuning":
        """
        Initialize `MoshiForFinetuning` from the original `LMModel`.
        """
        # Expose linear layer weights for DeepSpeed Zero-3 compatibility
        expose_linear_weights_for_zero3(moshi_lm)
        state_dict = moshi_lm.state_dict()
        device = next(moshi_lm.parameters()).device
        dtype = next(moshi_lm.parameters()).dtype

        # Clear the original model to save memory
        del moshi_lm

        # Initialize the new model
        moshi_lm = cls(device=device, dtype=dtype, **moshi_lm_kwargs).to(device=device, dtype=dtype)
        moshi_lm.load_state_dict(state_dict, strict=True)

        # Store the kwargs for the later use
        moshi_lm.moshi_lm_kwargs = moshi_lm_kwargs

        return moshi_lm

    def to_original_moshi_lm(self) -> LMModel:
        """
        Convert the model to the original `LMModel`.
        """
        state_dict = self.state_dict()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Convert the state dict to the original format
        state_dict = restore_linear_weights_from_exposed_state_dict(state_dict)

        # Initialize the original model
        moshi_lm = LMModel(device=device, dtype=dtype, **self.moshi_lm_kwargs).to(
            device=device, dtype=dtype
        )
        moshi_lm.load_state_dict(state_dict, strict=True)

        return moshi_lm

    def save_pretrained(self, save_dir: str):
        """
        Save the model to the given directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save the model
        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))
        # Save the kwargs
        with open(os.path.join(save_dir, "moshi_lm_kwargs.json"), "w") as f:
            json.dump(self.moshi_lm_kwargs, f, indent=4)

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MoshiForFinetuning":
        """
        Load the model from the given directory.
        """
        # Load the kwargs
        with open(os.path.join(save_dir, "moshi_lm_kwargs.json")) as f:
            moshi_lm_kwargs = json.load(f)
        # Drop kwargs that are not supported by the base Moshi model implementation.
        # These keys may appear when reusing kwargs files prepared for Moshi-LLaMA integrations.
        unsupported_keys = ("llama_name_or_path",)
        for key in unsupported_keys:
            if key in moshi_lm_kwargs:
                moshi_lm_kwargs.pop(key)
        # Initialize the model
        moshi_lm = cls(device=device, dtype=dtype, **moshi_lm_kwargs).to(device=device, dtype=dtype)
        # Load the model
        load_model(moshi_lm, os.path.join(save_dir, "model.safetensors"))

        moshi_lm.moshi_lm_kwargs = moshi_lm_kwargs

        return moshi_lm
    
class MoshiLlamaForFinetuning(MoshiLlama):
    """
    MoshiLM with Llama
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # DeepSpeed Zero-3 compatibility
        # Do not patch the transformer layers, as they are huggingface's Llama layers
        ## 1. Expose linear layer weights
        expose_linear_weights_for_zero3(self)
        ## 2. Apply patches for forward functions
        for layer in self.depformer.layers:
            for gating in layer.gating:
                gating.forward = activation_gating_forward.__get__(gating)
            layer.self_attn.forward = mha_forward.__get__(layer.self_attn)
        # Implement activation checkpointing
        self.depformer.activation_checkpointing = False
        self.depformer.forward = transformer_forward.__get__(self.depformer)

    def enable_activation_checkpointing(self, checkpointing_func):
        func = functools.partial(
            torch.utils.checkpoint.checkpoint,
            use_reentrant=True
        ) 
        self.transformer._set_gradient_checkpointing(
            enable=True, gradient_checkpointing_func=func#checkpointing_func
        )
        self.depformer.activation_checkpointing = True
        self.depformer.checkpointing_func = checkpointing_func

    def disable_activation_checkpointing(self):
        self.transformer._set_gradient_checkpointing(enable=False)
        self.depformer.activation_checkpointing = False

    @classmethod
    def from_original_moshi_llama(
        cls,
        moshi_llama: MoshiLlama,
        moshi_llama_kwargs: dict,
    ) -> "MoshiLlamaForFinetuning":
        """
        Initialize `MoshiLlamaForFinetuning` from the original `MoshiLlama`.
        """
        # Expose linear layer weights for DeepSpeed Zero-3 compatibility
        expose_linear_weights_for_zero3(moshi_llama)
        state_dict = moshi_llama.state_dict()
        device = next(moshi_llama.parameters()).device
        dtype = next(moshi_llama.parameters()).dtype

        # Clear the original model to save memory
        del moshi_llama

        # Initialize the new model
        moshi_llama_ft = cls(device=device, dtype=dtype, **moshi_llama_kwargs).to(device=device, dtype=dtype)
        moshi_llama_ft.load_state_dict(state_dict, strict=True)

        # Store the kwargs for the later use
        moshi_llama_ft.moshi_llama_kwargs = moshi_llama_kwargs

        return moshi_llama_ft

    def to_original_moshi_llama(self) -> MoshiLlama:
        """
        Convert the model to the original `MoshiLlama`.
        """
        state_dict = self.state_dict()
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Convert the state dict to the original format
        state_dict = restore_linear_weights_from_exposed_state_dict(state_dict)

        # Initialize the original model
        moshi_llama = MoshiLlama(device=device, dtype=dtype, **self.moshi_llama_kwargs).to(
            device=device, dtype=dtype
        )
        moshi_llama.load_state_dict(state_dict, strict=True)

        return moshi_llama

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MoshiLlamaForFinetuning":
        """
        Load the model from the given directory.
        """
        # Load the kwargs
        with open(os.path.join(save_dir, "moshi_llama_kwargs.json")) as f:
            moshi_llama_kwargs = json.load(f)
        # Initialize the model
        moshi_llama = cls(device=device, dtype=dtype, **moshi_llama_kwargs).to(device=device, dtype=dtype)
        # Load the model
        load_model(moshi_llama, os.path.join(save_dir, "model.safetensors"))
        moshi_llama.moshi_llama_kwargs = moshi_llama_kwargs
        return moshi_llama

    def save_pretrained(self, save_dir: str):
        """
        Save the model to the given directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save the model
        save_file(self.state_dict(), os.path.join(save_dir, "model.safetensors"))
        # Save the kwargs
        with open(os.path.join(save_dir, "moshi_llama_kwargs.json"), "w") as f:
            json.dump(self.moshi_llama_kwargs, f, indent=4)

class AutoMoshiForFinetuning:
    """
    Auto class for MoshiForFinetuning and MoshiLlamaForFinetuning.
    """
    @classmethod
    def from_pretrained(
        cls,
        save_dir: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16
    ) -> MoshiForFinetuning | MoshiLlamaForFinetuning:
        """
        Load the model from the given directory.
        """
        # get moshi_*_kwargs.json file
        kwargs_path = glob(os.path.join(save_dir, "moshi_*_kwargs.json"))[0]
        if kwargs_path.endswith("moshi_lm_kwargs.json"):
            return MoshiForFinetuning.from_pretrained(save_dir, device=device, dtype=dtype)
        elif kwargs_path.endswith("moshi_llama_kwargs.json"):
            return MoshiLlamaForFinetuning.from_pretrained(save_dir, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown kwargs file: {kwargs_path}. Expected moshi_lm_kwargs.json or moshi_llama_kwargs.json.")