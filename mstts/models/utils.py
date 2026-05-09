from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributions import constraints

from moshi.models import LMModel, loaders
from moshi.models.lm import ScaledEmbedding

def extend_moshi_modules_for_user_stream(lm: LMModel) -> LMModel:
    """
    Extend the depth transformer's modules to model user stream.
    0. (Optional) emb * 2
    1. depformer_in * 2
    2. depformer_emb * 2 + 1
    3. depformer (layers)
        3.1 self_attn.in_proj_weight * 2
        3.2 self_attn.out_proj * 2
        3.3 gating * 2
    4. linears * 2
    """
    lm_us = deepcopy(lm)

    # 0. emb (if the number of emb is 8, we extend it to 16)
    if len(lm_us.emb) == 8:
        lm_us.emb.extend(deepcopy(lm_us.emb))

    # 1. depformer_in
    lm_us.depformer_in.extend(deepcopy(lm.depformer_in))

    # 2. depformer_emb
    # depformer_emb doesn't have any embedding to encode the moshi's last acoustic token
    # (i.e., embedding for predicting user's semantic token) because moshi's semantic
    # token is predicted from text token, so we just use the first embedding in depformer_emb,
    # which is for predicting first acoustic token
    lm_us.depformer_emb.append(deepcopy(lm.depformer_emb[0]))
    lm_us.depformer_emb.extend(deepcopy(lm.depformer_emb))

    # 3. depformer (layers)
    for layer in lm_us.depformer.layers:
        # 3.1 self_attn.in_proj
        layer.self_attn.in_proj_weight.data = layer.self_attn.in_proj_weight.data.repeat(2, 1)

        # 3.2 self_attn.out_proj
        new_linear = torch.nn.Linear(
            in_features=layer.self_attn.out_proj.in_features,
            out_features=layer.self_attn.out_proj.out_features * 2,
            bias=False if layer.self_attn.out_proj.bias is None else True,
        )
        new_linear.load_state_dict(
            {
                "weight": layer.self_attn.out_proj.weight.repeat(2, 1),
            }
        )
        layer.self_attn.out_proj = new_linear

        # 3.3 gating
        layer.gating.extend(deepcopy(layer.gating))

    # 4. linears
    lm_us.linears.extend(deepcopy(lm_us.linears))

    return lm_us


def remove_moshi_modules_for_user_stream(lm_us: LMModel) -> LMModel:
    """
    Remove the depth transformer's modules to model user stream.
    Reverse of `extend_moshi_modules_for_user_stream()`.
    """
    lm = deepcopy(lm_us)
    device = next(lm.parameters()).device
    dtype = next(lm.parameters()).dtype

    # depformer_in
    lm.depformer_in = lm.depformer_in[:8]
    # depformer_emb
    lm.depformer_emb = lm.depformer_emb[:7]
    # depformer.layers
    for layer in lm.depformer.layers:
        # self_attn.in_proj_weight
        layer.self_attn.in_proj_weight.data = layer.self_attn.in_proj_weight.data[
            : layer.self_attn.in_proj_weight.shape[0] // 2
        ]
        # self_attn.out_proj
        out_proj = nn.Linear(
            in_features=layer.self_attn.out_proj.in_features,
            out_features=layer.self_attn.out_proj.out_features // 2,
            bias=False,
            device=device,
            dtype=dtype,
        )
        out_proj.load_state_dict(
            {
                "weight": layer.self_attn.out_proj.weight[
                    : layer.self_attn.out_proj.out_features // 2
                ]
            }
        )
        layer.self_attn.out_proj = out_proj
        # gating
        layer.gating = layer.gating[:8]
    # linears
    lm.linears = lm.linears[:8]

    return lm

@torch.no_grad()
def extend_embedding(embedding: ScaledEmbedding , num_new_vocab: int) -> torch.nn.Embedding:
    """
    Extend the embedding by adding new tokens.
    """
    num_old_vocab, dim = embedding.weight.shape
    old_weight_device = embedding.weight.device
    old_weight_dtype = embedding.weight.dtype

    old_weight = embedding.weight.to(torch.float32)
    mean = old_weight.mean(dim=0)
    covariance = ((old_weight - mean).T @ (old_weight - mean)) / num_old_vocab

    # Check if the covariance is positive definite.
    is_covariance_psd = constraints.positive_definite.check(covariance)
    if is_covariance_psd:
        # A distribution can be created and we can sample from it.
        dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mean.to(torch.float32),
            covariance_matrix=1e-9 * covariance.to(torch.float32),
        )
        new_weight_to_add = dist.sample(sample_shape=(num_new_vocab,)).to(old_weight_dtype)
    else:
        # just initialize with mean, because distribution will not be created
        new_weight_to_add = mean[None].repeat(num_new_vocab, 1).to(old_weight_dtype)

    new_weight = torch.cat([old_weight, new_weight_to_add], dim=0)
    
    new_embedding = ScaledEmbedding(
        num_old_vocab + num_new_vocab,
        dim,
        device=old_weight_device,
        dtype=old_weight_dtype,
        norm=embedding.norm,
        zero_idx=embedding.zero_idx,
    )
    new_embedding.load_state_dict({"weight": new_weight})
    return new_embedding

def add_speaker_embeddings(moshi_lm: loaders.LMModel, num_speakers: int) -> None:
    """
    Add speaker embeddings to the model.
    """
    num_new_text_embeddings = moshi_lm.text_card + 1 + num_speakers
    num_new_audio_embeddings = moshi_lm.card + 1 + num_speakers

    # 1 tempformer text embedding
    moshi_lm.text_emb = extend_embedding(moshi_lm.text_emb, num_speakers)
    assert moshi_lm.text_emb.num_embeddings == num_new_text_embeddings, \
        f"Expected {num_new_text_embeddings}, but got {moshi_lm.text_emb.num_embeddings}"

    # 2 tempformer audio embedding
    for i in range(len(moshi_lm.emb)):
        moshi_lm.emb[i] = extend_embedding(moshi_lm.emb[i], num_speakers)
        assert moshi_lm.emb[i].num_embeddings == num_new_audio_embeddings, \
            f"Expected {num_new_audio_embeddings}, but got {moshi_lm.emb[i].num_embeddings}"

    # 3 depformer text embedding
    moshi_lm.depformer_text_emb = extend_embedding(moshi_lm.depformer_text_emb, num_speakers)
    assert moshi_lm.depformer_text_emb.num_embeddings == num_new_text_embeddings, \
        f"Expected {num_new_text_embeddings}, but got {moshi_lm.depformer_text_emb.num_embeddings}"

    # 4 depformer audio embedding
    for i in range(len(moshi_lm.depformer_emb)):
        moshi_lm.depformer_emb[i] = extend_embedding(moshi_lm.depformer_emb[i], num_speakers)
        assert moshi_lm.depformer_emb[i].num_embeddings == num_new_audio_embeddings, \
            f"Expected {num_new_audio_embeddings}, but got {moshi_lm.depformer_emb[i].num_embeddings}"
