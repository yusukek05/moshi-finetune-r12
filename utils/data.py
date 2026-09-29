from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import BatchSampler


def main_speaker_streams(
    batched_examples: dict[str, list[Any]],
    speakers: list[str],
) -> list[np.ndarray]:
    """
    Pick the main speaker and create the streams.
    """
    list_of_streams = []
    for speaker in speakers:
        other = {"A": "B", "B": "A"}[speaker]
        for main_example, other_example in zip(
            batched_examples[speaker], batched_examples[other], strict=False
        ):
            streams = np.concat(
                [
                    np.array(main_example),  # 1 text + 8 audio
                    np.array(other_example)[1:],  # 8 audio
                ],
                axis=0,
            )
            list_of_streams.append(streams)
    return list_of_streams


def delay_and_pad_streams(
    list_of_streams: list[np.ndarray],
    delays: list[int],
    initial_token_ids: list[int],
    padding_token_ids: list[int],
) -> list[np.array]:
    """
    Apply delays and padding to the streams.
    """
    max_delay = max(delays)

    list_of_delayed_streams = []

    for streams in list_of_streams:
        num_streams, num_frames = streams.shape
        delayed = np.zeros((num_streams, num_frames + max_delay), dtype=streams.dtype)
        for i, delay in enumerate(delays):
            delayed[i, :delay] = initial_token_ids[i]
            delayed[i, delay : delay + num_frames] = streams[i]
            delayed[i, delay + num_frames :] = padding_token_ids[i]
        delayed = np.concat(
            [
                np.array(initial_token_ids, dtype=streams.dtype)[
                    :, None
                ],  # add initial token at the beginning
                delayed,
            ],
            axis=1,
        )

        list_of_delayed_streams.append(delayed)

    return list_of_delayed_streams


def split_streams(
    list_of_streams: list[np.ndarray],
    max_length: int,
) -> list[np.ndarray]:
    """
    Split the streams into chunks of max_length.
    """
    list_of_chunked_streams = []
    for streams in list_of_streams:
        num_streams, num_frames = streams.shape
        num_splits = -(-num_frames // max_length)
        list_of_chunked_streams.extend(np.array_split(streams, num_splits, axis=1))
    return list_of_chunked_streams


def filter_out_short_streams(
    list_of_streams: list[np.ndarray],
    min_length: int,
) -> list[np.ndarray]:
    """
    Filter out streams that are shorter than min_length.
    """
    return [s for s in list_of_streams if s.shape[1] >= min_length]


def make_streams_labels(
    list_of_streams: list[np.ndarray],
    initial_token_ids: list[int],
    zero_token_id: int,
) -> list[np.ndarray]:
    """
    Make the labels for the streams.
    """
    list_of_labels = []
    for streams in list_of_streams:
        num_streams, num_frames = streams.shape
        label = streams.copy()
        for i in range(num_streams):
            # zero out initial tokens
            label[i] = np.where(
                label[i] < initial_token_ids[i],
                label[i],
                zero_token_id,
            )
        list_of_labels.append(label)
    return list_of_labels


def preprocess_function(
    batched_examples: dict[str, list[Any]],
    speakers: list[str],
    max_length: int | None,
    min_length: int | None,
    delays: list[int],
    initial_token_ids: list[int],
    padding_token_ids: list[int],
    zero_token_id: int,
) -> dict[str, list[Any]]:
    # 1. make main speaker streams
    list_of_streams = main_speaker_streams(
        batched_examples=batched_examples,
        speakers=speakers,
    )

    # 2. delay and pad streams
    list_of_streams = delay_and_pad_streams(
        list_of_streams=list_of_streams,
        delays=delays,
        initial_token_ids=initial_token_ids,
        padding_token_ids=padding_token_ids,
    )

    # 3. split streams by max length
    if max_length is not None:
        list_of_streams = split_streams(
            list_of_streams=list_of_streams,
            max_length=max_length,
        )

    # 4. filter out short streams
    if min_length is not None:
        list_of_streams = filter_out_short_streams(
            list_of_streams=list_of_streams,
            min_length=min_length,
        )

    # 5. make labels
    list_of_labels = make_streams_labels(
        list_of_streams=list_of_streams,
        initial_token_ids=initial_token_ids,
        zero_token_id=zero_token_id,
    )

    list_of_num_streams = [streams.shape[0] for streams in list_of_streams]
    list_of_num_frames = [streams.shape[1] for streams in list_of_streams]

    return {
        "streams": list_of_streams,
        "labels": list_of_labels,
        "num_streams": list_of_num_streams,
        "num_frames": list_of_num_frames,
    }


def build_system_prompt_prefix(
    role_text_ids: list[int],
    voice_audio_codes: np.ndarray | None,
    num_streams: int,
    num_main_audio: int,
    text_padding_token_id: int,
    audio_padding_token_id: int,
    delimiter_text_id: int | None = None,
    dtype: np.dtype = np.int64,
) -> np.ndarray:
    """
    Build the *undelayed* PersonaPlex-style hybrid system-prompt prefix.

    Returns an array of shape (num_streams, prompt_len) laid out as the existing
    main-speaker streams (row 0 = text; rows [1, 1+num_main_audio) = agent/main
    audio; rows [1+num_main_audio, num_streams) = other/user audio).

    Segments, in order (voice precedes text, matching PersonaPlex so a text-only
    prompt can be prefilled when zero-shot voice cloning is not needed):
        [ voice prompt | text prompt | (delimiter) ]
      - voice prompt (len Tv): agent-audio rows <- voice_audio_codes,
        text row + user-audio rows <- padding.  Skipped when voice_audio_codes is None.
      - text prompt  (len Lt): text row <- role_text_ids, all audio rows <- padding.
      - delimiter    (len 1) : text row <- delimiter_text_id, audio rows <- padding.
        Skipped when delimiter_text_id is None.

    Every cell defaults to its per-stream padding token; only the active cells of
    each segment are overwritten.  The caller is responsible for masking the loss
    over this prefix (see preprocess_function_with_system_prompt).
    """
    assert num_streams >= 1 + num_main_audio, (
        f"num_streams {num_streams} < 1 + num_main_audio {num_main_audio}"
    )
    role_text_ids = list(role_text_ids)
    Lt = len(role_text_ids)
    if voice_audio_codes is not None:
        voice_audio_codes = np.asarray(voice_audio_codes)
        assert voice_audio_codes.ndim == 2, (
            f"voice_audio_codes must be 2D (num_main_audio, Tv), got {voice_audio_codes.ndim}D"
        )
        assert voice_audio_codes.shape[0] == num_main_audio, (
            f"voice_audio_codes rows {voice_audio_codes.shape[0]} != num_main_audio {num_main_audio}"
        )
        Tv = int(voice_audio_codes.shape[1])
    else:
        Tv = 0
    Ld = 1 if delimiter_text_id is not None else 0
    prompt_len = Tv + Lt + Ld
    assert prompt_len > 0, "empty system prompt (no voice, no text, no delimiter)"

    prefix = np.empty((num_streams, prompt_len), dtype=dtype)
    prefix[0, :] = text_padding_token_id
    prefix[1:, :] = audio_padding_token_id

    col = 0
    if Tv > 0:
        prefix[1 : 1 + num_main_audio, col : col + Tv] = voice_audio_codes.astype(dtype)
        col += Tv
    if Lt > 0:
        prefix[0, col : col + Lt] = np.array(role_text_ids, dtype=dtype)
        col += Lt
    if Ld > 0:
        prefix[0, col] = delimiter_text_id
        col += 1
    assert col == prompt_len
    return prefix


# ---------------------------------------------------------------------------
# PersonaPlex 論文 Fig.1 準拠の prefix ビルダー（0378 で追加）
#
# 論文 §3.1 と Fig.1 が示す構造を、既存実装との差分を明示して実装する:
#   - user 音声は全区間 440Hz サイン波   (既存: 音声パディング)
#   - agent 音声は「無音」               (既存: 音声パディング。パディングは
#                                          mimi の無音符号とは別物)
#   - text prompt を区切りトークンで両側から囲む (既存: 末尾に1個だけ)
#   - voice/text prompt の前後に Pause 区間  (既存: なし)
# 既存の build_system_prompt_prefix は残す（阿部さんの PoC 再現用）。
# ---------------------------------------------------------------------------
def build_system_prompt_prefix_v2(
    role_text_ids: list[int],
    voice_audio_codes: np.ndarray | None,
    num_streams: int,
    num_main_audio: int,
    text_padding_token_id: int,
    audio_padding_token_id: int,
    silence_codes: np.ndarray,
    sine_codes: np.ndarray,
    delimiter_text_id: int,
    pause_frames: int = 6,
    dtype: np.dtype = np.int64,
) -> np.ndarray:
    """論文 Fig.1 の Hybrid System Prompt を作る。

    レイアウト（列方向が時間）:
        [voice prompt][pause][DELIM role_text DELIM][pause]
      row 0                 : text   -> PAD / 区切り / role tokens
      rows 1..1+num_main_audio: agent -> 話者サンプル or 無音
      rows 1+num_main_audio..: user  -> 全区間 440Hz サイン波

    silence_codes / sine_codes は (num_main_audio, N) の定数トークン列。
    必要長に足りなければ時間方向に繰り返す。
    """
    assert num_streams >= 1 + num_main_audio
    n_user = num_streams - 1 - num_main_audio

    def tile(codes: np.ndarray, length: int) -> np.ndarray:
        codes = np.asarray(codes)
        assert codes.ndim == 2, f"codes must be 2D, got {codes.ndim}D"
        if length <= 0:
            return codes[:, :0]
        reps = -(-length // codes.shape[1])
        return np.tile(codes, (1, reps))[:, :length]

    role_text_ids = list(role_text_ids)
    Tv = int(np.asarray(voice_audio_codes).shape[1]) if voice_audio_codes is not None else 0
    Lt = len(role_text_ids)
    # 区切りは role text の両側
    text_seg = 1 + Lt + 1 if Lt > 0 else 0
    pause_a = pause_frames if Tv > 0 else 0          # voice と text の間
    pause_b = pause_frames                            # text と本編の間
    prompt_len = Tv + pause_a + text_seg + pause_b
    assert prompt_len > 0, "empty system prompt"

    prefix = np.empty((num_streams, prompt_len), dtype=dtype)
    prefix[0, :] = text_padding_token_id
    # agent 音声は既定で無音、user 音声は全区間サイン波
    prefix[1 : 1 + num_main_audio, :] = tile(silence_codes, prompt_len).astype(dtype)
    if n_user > 0:
        prefix[1 + num_main_audio :, :] = tile(sine_codes, prompt_len)[:n_user].astype(dtype)

    col = 0
    if Tv > 0:
        prefix[1 : 1 + num_main_audio, col : col + Tv] = np.asarray(voice_audio_codes, dtype=dtype)
        col += Tv + pause_a
    if Lt > 0:
        prefix[0, col] = delimiter_text_id
        prefix[0, col + 1 : col + 1 + Lt] = np.array(role_text_ids, dtype=dtype)
        prefix[0, col + 1 + Lt] = delimiter_text_id
        col += text_seg
    col += pause_b
    assert col == prompt_len, f"{col} != {prompt_len}"
    return prefix


def preprocess_function_with_system_prompt(
    batched_examples: dict[str, list[Any]],
    speakers: list[str],
    max_length: int | None,
    min_length: int | None,
    delays: list[int],
    initial_token_ids: list[int],
    padding_token_ids: list[int],
    zero_token_id: int,
    num_main_audio: int = 8,
    delimiter_text_id: int | None = None,
    prompt_text_key: str = "prompt_text_ids",
    prompt_audio_key: str = "prompt_audio",
) -> dict[str, list[Any]]:
    """
    Like `preprocess_function`, but prepends a PersonaPlex-style hybrid system
    prompt to each dialogue and masks the loss over the prompt region.

    Expects, in addition to the per-speaker dialogue streams ("A"/"B"), the
    per-example prompt fields:
      - prompt_text_ids: list[int]            (tokenized role/persona text)
      - prompt_audio    : [num_main_audio, Tv] (optional voice-prompt mimi codes; may be None)

    The conditioned agent is the main speaker; use speakers=["A"] (PersonaPlex
    conditions a single agent). Loss masking is done by setting the prompt region
    of `labels` to `zero_token_id`, which the loss in finetune.py already treats
    as ignore_index -- so no change to the training loop is required.
    """
    num_streams = len(initial_token_ids)
    text_pad = padding_token_ids[0]
    audio_pad = padding_token_ids[1]  # all audio rows share the same pad (= initial_token_id)
    num_examples = len(batched_examples[speakers[0]])

    # 1. dialogue streams (undelayed), same order as main_speaker_streams:
    #    outer loop over speakers, inner loop over examples.
    list_of_dialogue = main_speaker_streams(
        batched_examples=batched_examples,
        speakers=speakers,
    )

    # 2. prepend the hybrid system prompt to each dialogue (still undelayed)
    list_of_streams: list[np.ndarray] = []
    list_of_prompt_len: list[int] = []
    for s_idx in range(len(speakers)):
        for e_idx in range(num_examples):
            dialogue = list_of_dialogue[s_idx * num_examples + e_idx]
            role_text_ids = batched_examples[prompt_text_key][e_idx]
            voice = None
            if prompt_audio_key in batched_examples:
                v = batched_examples[prompt_audio_key][e_idx]
                if v is not None:
                    voice = np.asarray(v)
            prefix = build_system_prompt_prefix(
                role_text_ids=role_text_ids,
                voice_audio_codes=voice,
                num_streams=num_streams,
                num_main_audio=num_main_audio,
                text_padding_token_id=text_pad,
                audio_padding_token_id=audio_pad,
                delimiter_text_id=delimiter_text_id,
                dtype=dialogue.dtype,
            )
            combined = np.concatenate([prefix, dialogue], axis=1)
            # keep the whole prompt; truncate the dialogue tail if over budget
            if max_length is not None and combined.shape[1] > max_length:
                combined = combined[:, :max_length]
            list_of_streams.append(combined)
            list_of_prompt_len.append(prefix.shape[1])

    # 3. delay and pad (prepends one initial column + per-row delay)
    list_of_streams = delay_and_pad_streams(
        list_of_streams=list_of_streams,
        delays=delays,
        initial_token_ids=initial_token_ids,
        padding_token_ids=padding_token_ids,
    )

    # 4. labels (initial/pad tokens -> zero), then mask the prompt region
    list_of_labels = make_streams_labels(
        list_of_streams=list_of_streams,
        initial_token_ids=initial_token_ids,
        zero_token_id=zero_token_id,
    )
    masked_labels = []
    for label, prompt_len in zip(list_of_labels, list_of_prompt_len):
        label = label.copy()
        for i in range(label.shape[0]):
            # +1 for the initial column prepended by delay_and_pad_streams;
            # row i's content begins at column 1 + delays[i].
            end = min(1 + delays[i] + prompt_len, label.shape[1])
            label[i, :end] = zero_token_id
        masked_labels.append(label)

    # 5. filter out short streams (keep streams and labels aligned)
    if min_length is not None:
        keep = [k for k, s in enumerate(list_of_streams) if s.shape[1] >= min_length]
        list_of_streams = [list_of_streams[k] for k in keep]
        masked_labels = [masked_labels[k] for k in keep]
        list_of_prompt_len = [list_of_prompt_len[k] for k in keep]

    return {
        "streams": list_of_streams,
        "labels": masked_labels,
        "num_streams": [s.shape[0] for s in list_of_streams],
        "num_frames": [s.shape[1] for s in list_of_streams],
        "prompt_len": list_of_prompt_len,
    }


def preprocess_function_with_system_prompt_v2(
    batched_examples: dict[str, list[Any]],
    speakers: list[str],
    max_length: int | None,
    min_length: int | None,
    delays: list[int],
    initial_token_ids: list[int],
    padding_token_ids: list[int],
    zero_token_id: int,
    silence_codes: Any,
    sine_codes: Any,
    delimiter_text_id: int,
    num_main_audio: int = 8,
    pause_frames: int = 6,
    prompt_text_key: str = "prompt_text_ids",
    prompt_audio_key: str = "prompt_audio",
) -> dict[str, list[Any]]:
    """論文 Fig.1 準拠版。build_system_prompt_prefix_v2 を使う以外は
    preprocess_function_with_system_prompt と同一（loss マスクの取り方も同じ）。"""
    num_streams = len(initial_token_ids)
    text_pad = padding_token_ids[0]
    audio_pad = padding_token_ids[1]
    num_examples = len(batched_examples[speakers[0]])
    silence_codes = np.asarray(silence_codes)
    sine_codes = np.asarray(sine_codes)

    list_of_dialogue = main_speaker_streams(
        batched_examples=batched_examples, speakers=speakers
    )

    list_of_streams: list[np.ndarray] = []
    list_of_prompt_len: list[int] = []
    for s_idx in range(len(speakers)):
        for e_idx in range(num_examples):
            dialogue = list_of_dialogue[s_idx * num_examples + e_idx]
            role_text_ids = batched_examples[prompt_text_key][e_idx]
            voice = None
            if prompt_audio_key in batched_examples:
                v = batched_examples[prompt_audio_key][e_idx]
                if v is not None:
                    voice = np.asarray(v)
            prefix = build_system_prompt_prefix_v2(
                role_text_ids=role_text_ids,
                voice_audio_codes=voice,
                num_streams=num_streams,
                num_main_audio=num_main_audio,
                text_padding_token_id=text_pad,
                audio_padding_token_id=audio_pad,
                silence_codes=silence_codes,
                sine_codes=sine_codes,
                delimiter_text_id=delimiter_text_id,
                pause_frames=pause_frames,
                dtype=dialogue.dtype,
            )
            combined = np.concatenate([prefix, dialogue], axis=1)
            if max_length is not None and combined.shape[1] > max_length:
                combined = combined[:, :max_length]
            list_of_streams.append(combined)
            list_of_prompt_len.append(prefix.shape[1])

    list_of_streams = delay_and_pad_streams(
        list_of_streams=list_of_streams,
        delays=delays,
        initial_token_ids=initial_token_ids,
        padding_token_ids=padding_token_ids,
    )
    list_of_labels = make_streams_labels(
        list_of_streams=list_of_streams,
        initial_token_ids=initial_token_ids,
        zero_token_id=zero_token_id,
    )
    masked_labels = []
    for label, prompt_len in zip(list_of_labels, list_of_prompt_len):
        label = label.copy()
        for i in range(label.shape[0]):
            end = min(1 + delays[i] + prompt_len, label.shape[1])
            label[i, :end] = zero_token_id
        masked_labels.append(label)

    if min_length is not None:
        keep = [k for k, s in enumerate(list_of_streams) if s.shape[1] >= min_length]
        list_of_streams = [list_of_streams[k] for k in keep]
        masked_labels = [masked_labels[k] for k in keep]
        list_of_prompt_len = [list_of_prompt_len[k] for k in keep]

    return {
        "streams": list_of_streams,
        "labels": masked_labels,
        "num_streams": [s.shape[0] for s in list_of_streams],
        "num_frames": [s.shape[1] for s in list_of_streams],
        "prompt_len": list_of_prompt_len,
    }


def undelay_tokens(
    tokens: np.ndarray | torch.LongTensor, delays: list[int]
) -> np.ndarray | torch.LongTensor | None:
    """
    Restore the undelayed tokens from the delayed tokens

    Args:
        tokens (np.ndarray | torch.LongTensor): shape is (B, K, Td).
        delays (list[int]): delays for each of K codebooks

    Returns:
        undelayed_tokens (np.ndarray | torch.LongTensor): shape is (B, K, T).
            T is not necessarily equal to Td because of the delays.
    """
    max_delay = max(delays)
    B, K, Td = tokens.shape

    if Td < max_delay + 1:  # too short
        return None

    assert K == len(delays), f"Expected K == {len(delays)}, but got {K}"

    T = Td - max_delay
    if isinstance(tokens, np.ndarray):
        undelayed_tokens = np.zeros((B, K, T), dtype=tokens.dtype)
    else:
        undelayed_tokens = torch.zeros((B, K, T), dtype=tokens.dtype, device=tokens.device)
    for cb_index, delay in enumerate(delays):
        undelayed_tokens[:, cb_index] = tokens[:, cb_index, delay : delay + T]

    return undelayed_tokens


@dataclass
class Batch:
    example_ids: list[int] | None
    input_ids: torch.LongTensor
    text_attention_mask: torch.LongTensor
    labels: torch.LongTensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            example_ids=self.example_ids,
            input_ids=self.input_ids.to(device),
            text_attention_mask=self.text_attention_mask.to(device),
            labels=self.labels.to(device),
        )


class DataCollator:
    def __init__(self, zero_token_id: int):
        self.zero_token_id = zero_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> Batch:
        """
        Collate the examples into a batch.
        Args:
            examples (list[dict[str, Any]]): list of examples
                ```
                [
                    {
                        "streams": list[list[int]],
                        "labels": list[list[int]],
                        "num_streams": int,
                        "num_frames": int,
                    },
                    ...
                ]
                ```
        Returns:
            batch (Batch): batch of examples
        """
        # pad with zero tokens
        batch_size = len(examples)
        num_streams = examples[0]["num_streams"]
        max_frames = max([e["num_frames"] for e in examples])
        zero_ids = torch.full(
            (batch_size, num_streams, max_frames), fill_value=self.zero_token_id, dtype=torch.long
        )

        input_ids = zero_ids.clone()
        for i, e in enumerate(examples):
            input_ids[i, :, : e["num_frames"]] = torch.tensor(e["streams"], dtype=torch.long)

        text_attention_mask = zero_ids[:, 0, :].clone()  # (batch_size, max_frames)
        for i, e in enumerate(examples):
            text_attention_mask[i, : e["num_frames"]] = 1

        labels = zero_ids.clone()
        for i, e in enumerate(examples):
            labels[i, :, : e["num_frames"]] = torch.tensor(e["labels"], dtype=torch.long)

        example_ids = None
        if "example_id" in examples[0]:
            example_ids = [e["example_id"] for e in examples]

        batch = Batch(
            example_ids=example_ids,
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
        )
        return batch


# --- テキストのみバッチ（finetune_mono_text.py から移植） -------------------
# 全二重（2話者・user stream 有）の学習にテキストのみのバッチを混ぜるために、
# finetune_mono_text.py にあった機構を utils 側へ移した。移植元は単一話者専用
# （check_mono_args が moshi_speakers != ["A"] と model_user_stream を弾く）
# だったが、テキストバッチは音声ストリームを持たず text_emb -> transformer ->
# text_linear だけを通るので、話者数やユーザーストリームとは独立に使える。
#
# data_utils.py 側の Batch は kana_ids を持ち utils 側と非互換なので、
# クラスをそのまま import せず、utils の Batch に合わせて置き直している。


@dataclass
class TextBatch(Batch):
    """音声を持たないバッチ。forward の分岐で見分けるために型を分ける。"""

    def to(self, device: torch.device) -> "TextBatch":
        return TextBatch(
            example_ids=self.example_ids,
            input_ids=self.input_ids.to(device),
            text_attention_mask=self.text_attention_mask.to(device),
            labels=self.labels.to(device),
        )


class DataCollatorWithTextBatch(DataCollator):
    """音声バッチとテキストバッチのどちらでも受けられる collator。"""

    def __call__(self, examples: list[dict[str, Any]]) -> Batch | TextBatch:
        if examples[0].get("streams") is not None:
            return super().__call__(examples)
        # テキストのみ: (batch, seq_len) の 2 次元。長さを揃えて padding する
        streams = [e["text_stream"] for e in examples]
        max_len = max(len(s) for s in streams)
        input_ids = torch.full((len(streams), max_len), self.zero_token_id,
                               dtype=torch.long)
        mask = torch.zeros((len(streams), max_len), dtype=torch.long)
        for k, s in enumerate(streams):
            input_ids[k, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[k, : len(s)] = 1
        labels = input_ids.clone()
        # text prompt の領域を損失から外す（音声側が system prompt をマスクするのと同じ）。
        # zero_token_id は text_forward() の ignore_index かつ non_pad 判定の除外対象。
        # prompt_len が無い/0 のときは何もしない＝2026-09-02 までの動作。
        for k, e in enumerate(examples):
            n = int(e.get("prompt_len") or 0)
            if n > 0:
                labels[k, :n] = self.zero_token_id
        return TextBatch(
            example_ids=[e.get("example_id") for e in examples],
            input_ids=input_ids,
            text_attention_mask=mask,
            labels=labels,
        )


class AlternatingDatasetSampler(BatchSampler):
    """
    Sampling from two datasets alternatively in each batch.
    Args:
        base_dataset_indices (list[int]): The indices of the base dataset.
        alt_dataset_indices (list[int]): The indices of the alternative dataset.
        base_ratio (int): The number of batches to sample from the base dataset 
            before sampling one batch from the alternative dataset.
        base_batch_size (int): The batch size for the base dataset.
        alt_batch_size (int): The batch size for the alternative dataset.
    """
    def __init__(
            self,
            base_dataset_indices: list[int],
            alt_dataset_indices: list[int],
            base_ratio: int,
            batch_size: int,
            num_processes: int,
            drop_last: bool = False,
            seed: int = 0,
        ):
        self.base_indices = base_dataset_indices
        self.alt_indices = alt_dataset_indices
        self.base_ratio = base_ratio

        self.batch_size = batch_size
        self.num_processes = num_processes
        self.drop_last = drop_last

        self.seed = seed
        self.epoch = 0


        # calculate the number of global batches
        self.global_batch_size = self.batch_size * self.num_processes
        self.num_base_global_batches = len(self.base_indices) // self.global_batch_size
        if len(self.base_indices) % self.global_batch_size > 0 and not self.drop_last:
            self.num_base_global_batches += 1
        self.num_alt_global_batches = self.num_base_global_batches // self.base_ratio

    def set_epoch(self, epoch: int) -> None:
        """エポックごとに並びを変える。呼ばれないと全エポックで同じ順序になる。"""
        self.epoch = epoch

    def _shuffle_indices(self, indices: list[int]) -> list[int]:
        """
        Epoch-specific shuffling of indices.
        """
        rng = np.random.RandomState(self.seed + self.epoch)
        return rng.permutation(indices).tolist()
    
    def __len__(self) -> int:
        return (
            self.num_base_global_batches + self.num_alt_global_batches
        ) * self.num_processes

    def __iter__(self) -> Iterator[list[int]]:
        base_indices = self._shuffle_indices(self.base_indices)
        alt_indices = self._shuffle_indices(self.alt_indices)

        # adjust the number of base indices
        global_batch_size = self.batch_size * self.num_processes
        num_needed_base_indices = global_batch_size * self.num_base_global_batches
        if len(base_indices) < num_needed_base_indices:
            # fill the last batch with the first indices
            base_indices += base_indices[:num_needed_base_indices - len(base_indices)]
        elif len(base_indices) > num_needed_base_indices:
            base_indices = base_indices[:num_needed_base_indices]

        # adjust the number of alternative indices
        num_needed_alt_indices = global_batch_size * self.num_alt_global_batches
        if len(alt_indices) < num_needed_alt_indices:
            # repeat the alternative dataset until it has enough batches
            num_repeats = -(-num_needed_alt_indices // len(alt_indices))
            alt_indices = np.tile(alt_indices, num_repeats).tolist()
        if len(alt_indices) > num_needed_alt_indices:
            alt_indices = alt_indices[:num_needed_alt_indices]
        
        # split the global batches
        global_base_batches = np.array(base_indices).reshape(
            self.num_base_global_batches, global_batch_size
        ).tolist()
        global_alt_batches = np.array(alt_indices).reshape(
            self.num_alt_global_batches, global_batch_size
        ).tolist()

        # merge the base and alternative batches
        global_batches = []
        alt_global_batch_idx = 0
        for global_base_batch_idx, global_base_batch in enumerate(global_base_batches):
            global_batches.append(global_base_batch)
            if (global_base_batch_idx+1) % self.base_ratio == 0:
                global_batches.append(global_alt_batches[alt_global_batch_idx])
                alt_global_batch_idx += 1
        
        # yield batches
        for global_batch in global_batches:
            for process_id in range(self.num_processes):
                yield global_batch[process_id*self.batch_size:(process_id+1)*self.batch_size]
        
        self.epoch += 1
