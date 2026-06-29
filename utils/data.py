from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


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
