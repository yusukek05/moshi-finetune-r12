import re
from typing import Any, Optional, Iterator
from dataclasses import dataclass

import torch
import numpy as np
import sentencepiece as sp

from torch.utils.data import BatchSampler

import numpy as np

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
        for main_example, other_example in zip(batched_examples[speaker], batched_examples[other]):
            streams = np.concat([
                np.array(main_example), # 1 text + 8 audio
                np.array(other_example)[1:], # 8 audio
            ], axis=0)
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
            delayed[i, delay: delay + num_frames] = streams[i]
            delayed[i, delay + num_frames:] = padding_token_ids[i]
        delayed = np.concat([
            np.array(initial_token_ids, dtype=streams.dtype)[:, None], # add initial token at the beginning
            delayed,
        ], axis=1)

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
        list_of_chunked_streams.extend(
            np.array_split(streams, num_splits, axis=1)
        )
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
        max_length: Optional[int],
        min_length: Optional[int],
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

def random_delay_and_pad_streams(
        list_of_streams: list[np.ndarray],
        max_audio_delay: int,
        initial_token_ids: list[int],
        padding_token_ids: list[int],
        use_kana: bool = False,
    ) -> list[np.array]:
    """
    Apply *random* delays and padding to the streams.
    """

    list_of_delayed_streams = []
    for streams in list_of_streams:
        num_streams, num_frames = streams.shape
        audio_delay = np.random.randint(0, max_audio_delay+1)
        max_delay = audio_delay
        delays = []
        delays += [0]  # text stream delay
        if use_kana:
            delays += [0] # kana stream delay
        delays += [audio_delay] * (num_streams - int(1 + use_kana)) # audio streams delays

        delayed = np.zeros((num_streams, num_frames + max_delay), dtype=streams.dtype)
        for i, delay in enumerate(delays):
            delayed[i, :delay] = initial_token_ids[i]
            delayed[i, delay: delay + num_frames] = streams[i]
            delayed[i, delay + num_frames:] = padding_token_ids[i]
        delayed = np.concat([
            np.array(initial_token_ids, dtype=streams.dtype)[:, None], # add initial token at the beginning
            delayed,
        ], axis=1)

        list_of_delayed_streams.append(delayed)

    return list_of_delayed_streams

def group_streams_by_max_length(
        list_of_streams: list[np.ndarray],
        max_length: int,
    ) -> list[np.ndarray]:
    """
    Group the streams by max length.
    """
    concatenated_streams = np.concat(list_of_streams, axis=1)
    num_blocks = -(-concatenated_streams.shape[1] // max_length)
    list_of_block_streams = np.array_split(concatenated_streams, num_blocks, axis=1)
    return list_of_block_streams

def merge_text_audio_streams(text_ids: np.ndarray, audio_ids: np.ndarray, text_padding_id: int) -> np.ndarray:
    """
    Merge the tokenized text and audio stream of a single speaker.
    Args:
        text_ids: Tokenized text stream. Shape: [Kt, T_text]
        audio_ids: Tokenized audio stream. Shape: [Ka T_audio]
        text_padding_id: Padding id for text stream to fill the gap between audio and text streams.
    Returns:
        Merged tokenized text and audio stream. Shape: [K=Kt+Ka, T_audio]
    """
    assert text_ids.ndim == 2, f"Expected 2D tensor, got {text_ids.ndim}D tensor."
    assert audio_ids.ndim == 2, f"Expected 2D tensor, got {audio_ids.ndim}D tensor."
    # pad the text stream to match the audio stream
    text_len = text_ids.shape[1]
    audio_len = audio_ids.shape[1]
    if text_len > audio_len:
        text_ids = text_ids[:, :audio_len]
    elif text_len < audio_len:
        padding = np.full(
            (text_ids.shape[0], audio_len - text_len),
            fill_value=text_padding_id,
            dtype=text_ids.dtype
        )
        text_ids = np.concat([text_ids, padding], axis=1)
    # merge the streams
    merged_stream = np.concat([text_ids, audio_ids], axis=0)
    return merged_stream

def separate_kana_stream(
        list_of_streams: list[np.ndarray],
        kana_stream_index: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate the kana stream from the streams.
    """
    list_of_kana_stream = []
    for i in range(len(list_of_streams)):
        # extract kana stream
        list_of_kana_stream.append(
            list_of_streams[i][kana_stream_index]
        )
        # remove kana stream from the streams
        list_of_streams[i] = np.delete(
            list_of_streams[i],
            kana_stream_index,
            axis=0
        )
    return list_of_streams, list_of_kana_stream

def preprocess_function_for_singlechannel(
        batched_examples: dict[str, list[Any]],
        max_length: int,
        num_audio_codebooks: int,
        max_audio_delay: int,
        initial_token_ids: list[int],
        padding_token_ids: list[int],
        zero_token_id: int,
        use_kana: bool = False,
    ):
    assert num_audio_codebooks == len(padding_token_ids) - 1, \
        f"Expected {num_audio_codebooks} audio codebooks, but got {len(padding_token_ids) - 1}."
    assert len(initial_token_ids) == len(padding_token_ids), \
        f"Expected {len(padding_token_ids)} initial token ids, but got {len(initial_token_ids)}."

    # 1 merge single channel streams
    list_of_streams = []
    num_examples = len(batched_examples[f"A_text"])
    for i in range(num_examples):
        text_stream = np.array(batched_examples[f"A_text"][i])[None] # (1, T_t)
        if use_kana:
            kana_stream = np.array(batched_examples[f"A_kana"][i])[None] # (1, T_t)
            text_stream = np.concat([text_stream, kana_stream], axis=0) # (2, T_t)
        main_audio_streams = np.array(batched_examples[f"A_audio"][i])[:num_audio_codebooks]  # (K, T_a)
        merged_stream = merge_text_audio_streams(
            text_ids=text_stream,
            audio_ids=main_audio_streams,
            text_padding_id=padding_token_ids[0],
        )
        list_of_streams.append(merged_stream)

    # check the number of streams
    assert len(list_of_streams[0]) == len(initial_token_ids) + int(use_kana), \
        f"Expected {len(initial_token_ids) + int(use_kana)} streams, but got {len(list_of_streams[0])}."
    assert len(list_of_streams[0]) == len(padding_token_ids) + int(use_kana), \
        f"Expected {len(padding_token_ids) + int(use_kana)} streams, but got {len(list_of_streams[0])}."
    # handle kana stream
    if use_kana:
        initial_token_ids = (initial_token_ids[0:1] * 2) + initial_token_ids[1:]
        padding_token_ids = (padding_token_ids[0:1] * 2) + padding_token_ids[1:]

    # 2. random delay and pad streams
    list_of_streams = random_delay_and_pad_streams(
        list_of_streams=list_of_streams,
        max_audio_delay=max_audio_delay,
        initial_token_ids=initial_token_ids,
        padding_token_ids=padding_token_ids,
        use_kana=use_kana,
    )

    # 3. group streams by max length
    list_of_streams = group_streams_by_max_length(
        list_of_streams=list_of_streams,
        max_length=max_length,
    )

    # 4. make labels
    list_of_labels = make_streams_labels(
        list_of_streams=list_of_streams,
        initial_token_ids=initial_token_ids,
        zero_token_id=zero_token_id,
    )

    # 5. separate kana stream from the streams
    list_of_kana_stream = None
    if use_kana:
        list_of_streams, list_of_kana_stream = separate_kana_stream(
            list_of_streams=list_of_streams,
            kana_stream_index=1,  # kana stream is the second stream
        )
        list_of_labels, _ = separate_kana_stream(
            list_of_streams=list_of_labels,
            kana_stream_index=1,  # kana stream is the second stream
        )

    list_of_num_streams = [streams.shape[0] for streams in list_of_streams]
    list_of_num_frames = [streams.shape[1] for streams in list_of_streams]

    result = {
        "streams": list_of_streams,
        "labels": list_of_labels,
        "num_streams": list_of_num_streams,
        "num_frames": list_of_num_frames,
    }
    # `datasets.map(batched=True)` rejects None values; only emit the kana
    # stream when it is actually present.
    if list_of_kana_stream is not None:
        result["kana_stream"] = list_of_kana_stream
    return result

def merged_speaker_streams(
        batched_examples: dict[str, list[Any]],
        main_speaker: str,
        other_speaker: str,
        text_padding_token_id: int,
        end_of_text_padding_token_id: int,
        main_speaker_bos_id: int,
        other_speaker_bos_id: int,
    ) -> list[np.ndarray]:
    """
    Merge the both of two speakers' text streams for multi-stream TTS dataset.
    Returns:
        list_of_streams (list[np.ndarray]): list of merged streams.
            Each element is of shape (K+1, T).
        list_of_speaker_ids (list[tuple[int, int]]): list of speaker ids.
            Each element is a tuple of speaker ids for the main and the other speakers.
    """
    assert set([main_speaker, other_speaker]) == {"A", "B"}, \
        f"Expected speakers to be 'A' and 'B', but got {main_speaker} and {other_speaker}."
    padding_ids = [text_padding_token_id, end_of_text_padding_token_id]

    def fill_speaker_token_to_tail(text_tokens: list[int], speaker_id: int, overwrite_non_pad: bool) -> bool:
        """
        Fill the tail of the text tokens with the speaker token.
        e.g., [3, 3, 3, 3, 0, 8, 8, 8, 3, 0] -> [3, 3, 3, 3, 0, 8, 8, 8, 3, SPK1]
        """
        if not text_tokens:
            return False
        if text_tokens[-1] not in padding_ids and not overwrite_non_pad:
            return False

        if text_tokens[-2:] == [text_padding_token_id, end_of_text_padding_token_id]:
            text_tokens[-2:] = [end_of_text_padding_token_id, speaker_id]
        else:
            text_tokens[-1] = speaker_id
        return True
    
    def merge_text_tokens(text_tokens_main: list[int], text_tokens_other: list[int]) -> list[int]:
        last_speaker = None
        current_speaker = None
        merged_text_tokens = []
        num_frames = len(text_tokens_main)
        for i in range(num_frames):
            text_token_main = text_tokens_main[i]
            text_token_other = text_tokens_other[i]
            if text_token_main not in padding_ids:
                id = text_token_main
                current_speaker = main_speaker_bos_id
            elif text_token_other not in padding_ids:
                id = text_token_other
                current_speaker = other_speaker_bos_id
            else:
                if end_of_text_padding_token_id in [text_token_main, text_token_other]:
                    id = end_of_text_padding_token_id
                else:
                    id = text_padding_token_id
            if current_speaker is not None and last_speaker != current_speaker:
                # switch speaker
                # if last_speaker is not None: # dont need to fill eos if it is the initial speech
                #     # add eos toekn to the tail of the previous speaker
                #     fill_eos_to_tail_of_last_speaker(merged_text_tokens)
                # add bos token to the tail of the current speaker
                success = fill_speaker_token_to_tail(
                    merged_text_tokens,
                    speaker_id=current_speaker,
                    overwrite_non_pad=current_speaker==main_speaker_bos_id
                )
                if success:
                    last_speaker = current_speaker
                else:
                    # dont switch speaker if the bos token cannot be added
                    id = text_token_main
                    current_speaker = last_speaker
            merged_text_tokens.append(id)
        return merged_text_tokens

    list_of_streams = []
    for main_example, other_example in zip(batched_examples[main_speaker], batched_examples[other_speaker]):
        text_stream = np.array(merge_text_tokens(main_example[0], other_example[0])) # (T,)
        audio_streams = np.concat( # (K, T)
            [np.array(main_example)[1:], np.array(other_example)[1:]],
            axis=0
        )
        list_of_streams.append(
            np.concat([text_stream[None], audio_streams], axis=0) # (K+1, T)
        )
    
    return list_of_streams

def inject_speaker_embedding_ids(
        list_of_streams: list[np.ndarray | torch.LongTensor],
        list_of_speaker_ids: list[tuple[int, int]],
        speaker_id_to_embedding_id: dict[str, list[int]],
        num_speaker_embedding_frames: int,
    ) -> list[np.ndarray | torch.LongTensor]:
    """
    Inject the speaker embedding id to the streams as prefix.
    """
    list_of_injected_streams = []
    for streams, speaker_ids in zip(list_of_streams, list_of_speaker_ids):
        pt = isinstance(streams, torch.Tensor)
        if pt:
            device = streams.device
            streams = streams.cpu().numpy()

        main_speaker_text_emb_id = speaker_id_to_embedding_id["text"][speaker_ids[0]]
        other_speaker_text_emb_id = speaker_id_to_embedding_id["text"][speaker_ids[1]]
        main_speaker_audio_emb_id = speaker_id_to_embedding_id["audio"][speaker_ids[0]]
        other_speaker_audio_emb_id = speaker_id_to_embedding_id["audio"][speaker_ids[1]]

        emb_ids = np.zeros((streams.shape[0], num_speaker_embedding_frames*2), dtype=streams.dtype)
        emb_ids[0, :num_speaker_embedding_frames] = main_speaker_text_emb_id
        emb_ids[0, num_speaker_embedding_frames:] = other_speaker_text_emb_id
        emb_ids[1:, :num_speaker_embedding_frames] = main_speaker_audio_emb_id
        emb_ids[1:, num_speaker_embedding_frames:] = other_speaker_audio_emb_id

        streams = np.concat([emb_ids, streams], axis=-1)
        if pt:
            streams = torch.tensor(streams, dtype=torch.long, device=device)
        list_of_injected_streams.append(streams)

    return list_of_injected_streams

def preprocess_function_for_multistream_tts(
        batched_examples: dict[str, list[Any]],
        speakers: list[str],
        max_length: Optional[int],
        min_length: Optional[int],
        delays: list[int],
        initial_token_ids: list[int],
        padding_token_ids: list[int],
        end_of_text_padding_token_id: int,
        main_speaker_bos_id: int,
        other_speaker_bos_id: int,
        zero_token_id: int,
    ) -> dict[str, list[Any]]:
    # 1. make merged speaker streams
    list_of_streams = []
    for main_speaker in speakers:
        other_speaker = {"A": "B", "B": "A"}[main_speaker]
        list_of_streams_ = merged_speaker_streams(
            batched_examples=batched_examples,
            main_speaker=main_speaker,
            other_speaker=other_speaker,
            text_padding_token_id=padding_token_ids[0],
            end_of_text_padding_token_id=end_of_text_padding_token_id,
            main_speaker_bos_id=main_speaker_bos_id,
            other_speaker_bos_id=other_speaker_bos_id,
        )
        list_of_streams.extend(list_of_streams_)

    # 2. split streams by max length
    if max_length is not None:
        list_of_streams = split_streams(
            list_of_streams=list_of_streams, max_length=max_length,
        )

    # 3. filter out short streams
    if min_length is not None:
        list_of_streams = filter_out_short_streams(
            list_of_streams=list_of_streams, min_length=min_length,
        )

    # 4. delay and pad streams
    list_of_streams = delay_and_pad_streams(
        list_of_streams=list_of_streams,
        delays=delays,
        initial_token_ids=initial_token_ids,
        padding_token_ids=padding_token_ids,
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

symbols = str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})
def tokenize_text_chat_for_multistream_tts(
        text_chat: list[str],
        text_tokenizer: sp.SentencePieceProcessor,
        main_speaker_bos_id: int,
        other_speaker_bos_id: int,
        main_speaker_first: bool = True,
    ) -> list[str]:
    """
    Tokenize the text chats for multi-stream TTS.
    """
    def normalize_and_tokenize_sentence(sentence: str) -> list[int]:
        # 1 Normalize the text
        sentence = sentence.strip()
        # 1.1 lower
        sentence = sentence.lower()
        # 1.2 normalize symbols
        sentence = sentence.translate(symbols)
        # 1.3 add period at the end if it is not there
        if not sentence.endswith(("。", "、", "!", "?")):
            sentence += "。"

        # 2 Tokenize the text
        token_list = text_tokenizer.encode_as_pieces(sentence)
        if not token_list:
            return []
        # remove the underline token
        piece_underline = "▁"
        if token_list[0] == piece_underline:
            token_list = token_list[1:]
        elif token_list[0].startswith(piece_underline):
            token_list[0] = token_list[0][1:]
        return text_tokenizer.piece_to_id(token_list)

    token_ids = []
    turn_id_to_speaker_bos_id = [main_speaker_bos_id, other_speaker_bos_id]
    if not main_speaker_first:
        turn_id_to_speaker_bos_id = turn_id_to_speaker_bos_id[::-1]
    for i, sentence in enumerate(text_chat):
        bos_id = turn_id_to_speaker_bos_id[i % 2]
        ids = normalize_and_tokenize_sentence(sentence)
        token_ids.extend([bos_id] + ids)
    return token_ids


def undelay_tokens(tokens: np.ndarray | torch.LongTensor, delays: list[int]) ->Optional[np.ndarray | torch.LongTensor]:
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

    if Td < max_delay+1: # too short
        return None

    assert K == len(delays), \
        f"Expected K == {len(delays)}, but got {K}"
    
    T = Td - max_delay
    if isinstance(tokens, np.ndarray):
        undelayed_tokens = np.zeros((B, K, T), dtype=tokens.dtype)
    else:
        undelayed_tokens = torch.zeros((B, K, T), dtype=tokens.dtype, device=tokens.device)
    for cb_index, delay in enumerate(delays):
        undelayed_tokens[:, cb_index] = tokens[:, cb_index, delay:delay+T]

    return undelayed_tokens


def find_last_non_padding(tokens: np.ndarray | torch.LongTensor, buffer_len: int) -> int:
    """
    Find the last non-padding token in the tokens tensor.
    Args:
        tokens (np.ndarray | torch.LongTensor): Array of shape (T,) where T is the length of the streams.
        buffer_len (int): Length of the buffer to check for padding tokens.
    Returns:
        int: The index of the last non-padding token in the stream.
    """
    assert len(tokens.shape) == 1, f"Tokens must be a 1D (T) tensor, got {len(tokens.shape)}D tensor"
    assert buffer_len > 0, "Buffer length must be greater than 0"
    padding_token_id = 3
    if (tokens[-buffer_len:] != padding_token_id).any():
        return tokens.shape[0]
    for i in reversed(range(tokens.shape[0])):
        if i < buffer_len:
            break
        if tokens[i - buffer_len] != padding_token_id:
            return i
    return i


@dataclass
class Batch:
    example_ids: list[int] | None
    input_ids: torch.LongTensor
    kana_ids: torch.LongTensor | None
    text_attention_mask: torch.LongTensor
    labels: torch.LongTensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            example_ids=self.example_ids,
            input_ids=self.input_ids.to(device),
            kana_ids=self.kana_ids.to(device) if self.kana_ids is not None else None,
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
            (batch_size, num_streams, max_frames),
            fill_value=self.zero_token_id,
            dtype=torch.long
        )

        input_ids = zero_ids.clone()
        for i, e in enumerate(examples):
            input_ids[i, :, :e["num_frames"]] = torch.tensor(e["streams"], dtype=torch.long)

        text_attention_mask = zero_ids[:, 0, :].clone() # (batch_size, max_frames)
        for i, e in enumerate(examples):
            text_attention_mask[i, :e["num_frames"]] = 1

        labels = zero_ids.clone()
        for i, e in enumerate(examples):
            labels[i, :, :e["num_frames"]] = torch.tensor(e["labels"], dtype=torch.long)

        example_ids = None
        if "example_id" in examples[0]:
            example_ids = [e["example_id"] for e in examples]

        # check if kana stream is present
        kana_ids = None
        kana_stream = examples[0].get("kana_stream")
        if kana_stream is not None:
            kana_ids = torch.full(
                (batch_size, max_frames),
                fill_value=self.zero_token_id,
                dtype=torch.long
            )
            for i, e in enumerate(examples):
                kana_ids[i, :e["num_frames"]] = torch.tensor(e["kana_stream"], dtype=torch.long)

        batch = Batch(
            example_ids=example_ids,
            input_ids=input_ids,
            kana_ids=kana_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
        )
        return batch

@dataclass
class TextBatch(Batch):
    def to(self, device: torch.device) -> "TextBatch":
        return TextBatch(
            example_ids=self.example_ids,
            input_ids=self.input_ids.to(device),
            text_attention_mask=self.text_attention_mask.to(device),
            labels=self.labels.to(device),
        )

class DataCollatorWithTextBatch(DataCollator):
    def __init__(self, zero_token_id: int):
        self.zero_token_id = zero_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> Batch | TextBatch:
        """
        Collate the examples into a batch with text streams.
        Args:
            examples (list[dict[str, Any]]): list of examples
                ```
                [
                    {
                        "text_stream": list[int],
                        "example_id": int,
                    },
                    ...
                ]
                ```
        """
        # check batch type
        if examples[0]["streams"] is not None:
            assert all([e["streams"] is not None for e in examples]), \
                "All examples should have the same type of streams."
            batch_type = "audio"
        elif examples[0]["text_stream"] is not None:
            assert all([e["text_stream"] is not None for e in examples]), \
                "All examples should have the same type of text stream."
            batch_type = "text"
        else:
            raise ValueError(f"Unknown batch type: {examples[0]}")
        
        if batch_type == "audio":
            return super().__call__(examples)
        
        # pad with zero tokens
        batch_size = len(examples)
        max_length = max([len(e["text_stream"]) for e in examples])
        zero_ids = torch.full(
            (batch_size, max_length),
            fill_value=self.zero_token_id,
            dtype=torch.long
        )

        input_ids = zero_ids.clone()
        for i, e in enumerate(examples):
            input_ids[i, :len(e["text_stream"])] = torch.tensor(e["text_stream"], dtype=torch.long)

        labels = input_ids.clone()

        text_attention_mask = zero_ids.clone()
        for i, e in enumerate(examples):
            text_attention_mask[i, :len(e["text_stream"])] = 1

        example_ids = None
        if "example_id" in examples[0]:
            example_ids = [e["example_id"] for e in examples]

        batch = TextBatch(
            example_ids=example_ids,
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
        )
        return batch

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
