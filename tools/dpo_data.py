#!/usr/bin/env python3
"""DPO data path + sequence log-prob for Moshi multistream continuation.

The trainer needs, for a (context, continuation) pair, log pi(continuation |
context) under the model. Generation (`generate.py`) saved each continuation as
an UNDELAYED (1+K, T) array (row 0 = text, rows 1..K = audio), the same row
order as `utils.data.main_speaker_streams`. To score it under the model we must
put the (prompt ++ continuation) back into the delayed training layout and sum
the token log-probs over the CONTINUATION region only.

Reconstruction (reuses the tested delay helpers; delay/undelay are inverse up to
the max_delay=1 boundary frame, applied identically to chosen/rejected and
policy/ref so the DPO objective stays consistent):

    prompt_undelayed  = undelay(example.streams[:, 1:])[:, :prompt_len]   # from parquet
    full_undelayed    = concat([prompt_undelayed, continuation], axis=1)
    delayed           = delay_and_pad_streams([full_undelayed])           # training layout
    labels            = make_streams_labels(delayed); mask prompt region -> zero (ignore)

`sequence_logprob` then mirrors finetune.py's tempformer/depformer forward but
returns, per example, the summed log-prob over non-ignored text + audio targets.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from utils import (
    Batch,
    DataCollator,
    undelay_tokens,
)
from utils.data import (
    delay_and_pad_streams,
    make_streams_labels,
)


def token_id_config(moshi_lm):
    """Delays + per-stream initial/padding/zero ids, exactly as generate.py builds them."""
    delays = list(moshi_lm.delays)
    initial_token_ids = [moshi_lm.text_initial_token_id] + [
        moshi_lm.initial_token_id
    ] * moshi_lm.num_audio_codebooks
    padding_token_ids = [moshi_lm.text_padding_token_id] + [
        moshi_lm.initial_token_id
    ] * moshi_lm.num_audio_codebooks
    return {
        "delays": delays,
        "initial_token_ids": initial_token_ids,
        "padding_token_ids": padding_token_ids,
        "zero_token_id": moshi_lm.zero_token_id,
    }


def prompt_undelayed_from_streams(delayed_streams: np.ndarray, delays: list[int],
                                  prompt_len: int) -> np.ndarray:
    """Recover the first `prompt_len` undelayed frames from a preprocessed
    (delayed, incl. initial column) example — drop the initial column, undelay."""
    tokens = np.asarray(delayed_streams)[:, 1:][None]  # (1, K, N+max_delay)
    und = undelay_tokens(tokens, delays)
    if und is None:
        return None
    return und[0][:, :prompt_len]


def build_example(prompt_undelayed: np.ndarray, continuation: np.ndarray,
                  cfg: dict) -> dict:
    """(prompt_undelayed, continuation_undelayed) -> preprocessed example dict
    (streams + labels) with the prompt region masked out (labels -> zero)."""
    delays = cfg["delays"]
    prompt_len = prompt_undelayed.shape[1]
    full = np.concatenate([prompt_undelayed, np.asarray(continuation)], axis=1)
    delayed = delay_and_pad_streams([full], delays, cfg["initial_token_ids"],
                                    cfg["padding_token_ids"])[0]
    labels = make_streams_labels([delayed], cfg["initial_token_ids"],
                                 cfg["zero_token_id"])[0]
    # mask the prompt region: row i content starts at col (1 + delays[i]); mask
    # everything through the end of the prompt so only the continuation scores.
    labels = labels.copy()
    for i in range(labels.shape[0]):
        end = min(1 + delays[i] + prompt_len, labels.shape[1])
        labels[i, :end] = cfg["zero_token_id"]
    return {
        "streams": delayed,
        "labels": labels,
        "num_streams": delayed.shape[0],
        "num_frames": delayed.shape[1],
    }


def collate(examples: list[dict], zero_token_id: int) -> Batch:
    return DataCollator(zero_token_id=zero_token_id)(examples)


@torch.no_grad()
def _noop():
    pass


def sequence_logprob(moshi_lm, batch: Batch, model_user_stream: bool,
                     length_normalize: bool = True) -> torch.Tensor:
    """Per-example log-prob over the non-ignored text + audio targets.

    Mirrors finetune.py tempformer_forward + depformer_forward, but returns
    log-probs (sum of -CE at scored positions) instead of the SFT loss reductions.
    Returns a (B,) tensor. Positions whose label == zero_token_id contribute 0
    (F.cross_entropy ignore_index), i.e. prompt/initial/pad-collate are excluded.

    length_normalize=True divides each sequence's log-prob by its number of
    scored (text + audio) tokens -> per-token mean log-prob. Essential here:
    a continuation is ~17 streams x ~250 frames ~ 4k tokens, so the *summed*
    log-prob margin is O(1e3) and saturates the DPO sigmoid for any usable beta;
    it also removes DPO's length bias. Set False for the raw summed log-prob.
    """
    zero = moshi_lm.zero_token_id

    # ---- text (tempformer) ----
    text_emb = moshi_lm.text_emb(batch.input_ids[:, 0])
    audio_emb = None
    for acb in range(moshi_lm.num_audio_codebooks):
        e = moshi_lm.emb[acb](batch.input_ids[:, moshi_lm.audio_offset + acb])
        audio_emb = e if audio_emb is None else audio_emb + e
    tempformer_out = moshi_lm.transformer(text_emb + audio_emb,
                                          attention_mask=batch.text_attention_mask)
    if moshi_lm.out_norm:
        tempformer_out = moshi_lm.out_norm(tempformer_out)
    text_logits = moshi_lm.text_linear(tempformer_out).float()
    text_logits = text_logits[..., :-1, :].contiguous()
    text_labels = batch.labels[:, 0, 1:].contiguous()
    text_ce = F.cross_entropy(
        text_logits.view(-1, moshi_lm.text_card), text_labels.view(-1),
        ignore_index=zero, reduction="none",
    ).view(text_labels.size())
    logp_text = -text_ce.sum(dim=1)  # (B,)

    # ---- audio (depformer) ----
    depformer_inputs = []
    for acb in range(moshi_lm.dep_q):
        lin = moshi_lm.depformer_in[acb] if moshi_lm.depformer_multi_linear else moshi_lm.depformer_in[0]
        depformer_inputs.append(lin(tempformer_out[:, :-1]))
    depformer_input = torch.stack(depformer_inputs, dim=2)

    last = [moshi_lm.depformer_text_emb(batch.input_ids[:, 0, 1:])]
    for acb in range(moshi_lm.dep_q - 1):
        last.append(moshi_lm.depformer_emb[acb](batch.input_ids[:, moshi_lm.audio_offset + acb, 1:]))
    depformer_input = depformer_input + torch.stack(last, dim=2)
    depformer_input = torch.flatten(depformer_input, 0, 1)
    depformer_out = moshi_lm.depformer(depformer_input)

    audio_logits = []
    for acb in range(moshi_lm.dep_q):
        lin = moshi_lm.linears[acb] if moshi_lm.depformer_multi_linear else moshi_lm.linears[0]
        audio_logits.append(lin(depformer_out[:, acb]))
    audio_logits = torch.stack(audio_logits, dim=1).float()  # (B*T, dep_q, card)

    if model_user_stream:
        audio_labels = batch.labels[:, 1:, 1:].transpose(1, 2).contiguous()
    else:
        audio_labels = batch.labels[:, 1:9, 1:].transpose(1, 2).contiguous()
    audio_ce = F.cross_entropy(
        audio_logits.view(-1, moshi_lm.card), audio_labels.view(-1),
        ignore_index=zero, reduction="none",
    ).view(audio_labels.size())  # (B, T, dep_q)
    logp_audio = -audio_ce.sum(dim=(1, 2))  # (B,)

    logp = logp_text + logp_audio
    if length_normalize:
        n_text = (text_labels != zero).sum(dim=1)  # (B,)
        n_audio = (audio_labels != zero).sum(dim=(1, 2))  # (B,)
        n_scored = (n_text + n_audio).clamp(min=1).to(logp.dtype)
        logp = logp / n_scored
    return logp
