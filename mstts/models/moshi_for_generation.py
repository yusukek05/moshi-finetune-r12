from typing import Any

import torch
from moshi.models import LMModel
from moshi.utils.compile import CUDAGraphed
from moshi.utils.sampling import sample_token

from models.moshi_for_finetuning import (
    MoshiForFinetuning,
    MoshiLlamaForFinetuning,
)


class MoshiForConditionalGeneration:
    def __init__(self, moshi_lm: LMModel | MoshiForFinetuning | MoshiLlamaForFinetuning):
        self.moshi_lm = moshi_lm

    def prepare_generation(
        self,
        batch_size: int,
        text_sampling_params: dict[str, Any],
        audio_sampling_params: dict[str, Any],
    ) -> None:
        self.text_sampling_params = text_sampling_params
        self.audio_sampling_params = audio_sampling_params
        if isinstance(self.moshi_lm, MoshiLlamaForFinetuning):
            # Cuda graphing is not supported for MoshiLlamaForFinetuning
            self.tempformer_forward_graphed = None
        else:
            self.tempformer_forward_graphed = CUDAGraphed(self.moshi_lm.forward_text)
        self.depformer_step_graphed = CUDAGraphed(self.depformer_step)
        self.moshi_lm.streaming_forever(batch_size)

    def finish_generation(self):
        self.text_sampling_params = None
        self.audio_sampling_params = None
        self.tempformer_forward_graphed = None
        self.depformer_step_graphed = None
        self.moshi_lm.reset_streaming()

    def depformer_step(
        self, text_token: torch.LongTensor, transformer_out: torch.Tensor
    ) -> torch.LongTensor:
        """
        Generate next audio tokens from the given text token.

        Args:
            text_token (torch.Tensor):
                The text token to generate from. shape is [B]
            transformer_out (torch.Tensor):
                The output of the transformer. shape is [B, 1, dim]

        Returns:
            audio_tokens (torch.Tensor):
                The generated audio tokens. shape is [B, K]
        """
        (B,) = text_token.shape
        prev_token = text_token
        depformer_tokens: list[torch.Tensor] = []
        assert not self.moshi_lm.depformer.is_streaming
        with self.moshi_lm.depformer.streaming(B):
            for cb_index in range(self.moshi_lm.dep_q):
                input_ = prev_token[:, None, None]
                logits = self.moshi_lm.forward_depformer(cb_index, input_, transformer_out)
                next_token = sample_token(logits.float(), **self.audio_sampling_params)
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]  # shape is B
                depformer_tokens.append(next_token)
                prev_token = next_token

        assert len(depformer_tokens) == self.moshi_lm.dep_q, (
            len(depformer_tokens),
            self.moshi_lm.dep_q,
        )
        out = torch.stack(depformer_tokens, dim=1)
        assert out.shape == (B, self.moshi_lm.dep_q), out.shape
        return out

    @torch.no_grad()
    def step(self, last_tokens: torch.LongTensor) -> torch.LongTensor:
        """
        Generate next tokens from the given last tokens.

        Args:
            last_tokens (torch.LongTensor):
                The last tokens to generate from.
                shape is [B, K, T], where B is the batch size, K is the number of codebooks, and T is the number of tokens.
                T > 1 means it is the first step of generation when prompt tokens are given.

        Returns:
            sampled_tokens (torch.LongTensor):
                The sampled tokens. shape is [B, K, 1]
        """

        if not self.moshi_lm.transformer.is_streaming:
            raise RuntimeError("moshi_lm.transformer should be in streaming mode")

        B, K, T = last_tokens.shape
        assert K == self.moshi_lm.num_codebooks, (
            f"Expected {self.moshi_lm.num_codebooks}, but got {K}"
        )

        # 1 tempral transformer forward
        if T == 1 and self.tempformer_forward_graphed is not None:
            temp_out, text_logits = self.tempformer_forward_graphed(last_tokens)
        else:
            # cannot forward multiple tokens to the graphed function
            temp_out, text_logits = self.moshi_lm.forward_text(last_tokens)

        assert temp_out.shape == (
            B,
            T,
            self.moshi_lm.dim,
        ), f"Expected shape {(B, T, self.moshi_lm.dim)}, but got {temp_out.shape}"
        assert text_logits.shape == (
            B,
            1,
            T,
            self.moshi_lm.text_card,
        ), f"Expected shape {(B, 1, T, self.moshi_lm.text_card)}, but got {text_logits.shape}"

        temp_out = temp_out[:, -1][:, None]  # shape is [B, 1, dim]
        text_logits = text_logits[:, :, -1]  # shape is [B, 1, text_card]

        # 2 sample next text tokens
        text_token = sample_token(text_logits.float(), **self.text_sampling_params)
        assert text_token.shape == (B, 1), f"Expected shape {(B, 1)}, but got {text_token.shape}"
        text_token = text_token[:, 0]  # shape is [B]

        # 3 sample next audio tokens
        audio_tokens = self.depformer_step_graphed(text_token, temp_out)
        assert audio_tokens.shape == (B, self.moshi_lm.dep_q), audio_tokens.shape

        sampled_tokens = torch.cat([text_token[:, None], audio_tokens], dim=-1)
        assert sampled_tokens.shape == (B, self.moshi_lm.num_codebooks), sampled_tokens.shape
        sampled_tokens = sampled_tokens[..., None]  # shape is [B, num_codebooks, 1]

        return sampled_tokens

    def generate(
        self,
        prompt_tokens: torch.LongTensor,
        generation_length: int,
        text_sampling_params: dict[str, Any],
        audio_sampling_params: dict[str, Any],
    ) -> torch.LongTensor:
        """
        Generate text and audio streams from the given prompt tokens.
        Make sure that prompt and output tokens are delayed by moshi_lm.delays.

        Args:
            prompt_tokens (torch.LongTensor):
                The prompt tokens to generate from.
                shape is [B, K, Tp], where B is the batch size, K is the number of codebooks, and Tp is the number of tokens.
            generation_length (int):
                The number of tokens to generate.
            text_sampling_params (dict[str, Any]):
                The sampling parameters for text tokens.
            audio_sampling_params (dict[str, Any]):
                The sampling parameters for audio tokens.

        Returns:
            generated_tokens (torch.LongTensor):
                The generated tokens. shape is [B, K, Tg]
        """

        B, K, Tp = prompt_tokens.shape
        assert K == self.moshi_lm.num_codebooks, (
            f"Expected {self.moshi_lm.num_codebooks}, but got {K}"
        )

        # prepare generation
        self.prepare_generation(B, text_sampling_params, audio_sampling_params)

        # generate
        list_of_tokens = []
        last_tokens = prompt_tokens
        for _ in range(generation_length):
            last_tokens = self.step(last_tokens)
            list_of_tokens.append(last_tokens)

        # end generation
        self.finish_generation()

        generated_tokens = torch.cat(list_of_tokens, dim=-1)
        assert generated_tokens.shape == (B, K, generation_length), generated_tokens.shape

        return generated_tokens


class MoshiForMultiStreamTTS(MoshiForConditionalGeneration):
    @torch.no_grad()
    def step(
        self,
        last_tokens: torch.LongTensor,
        next_text_tokens: torch.LongTensor,
        force_text_token: torch.BoolTensor,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Almost same as MoshiForConditionalGeneration.step, but for multi-stream TTS.
        Specifically, if a sampled text token is the padding token (0 or 3), it is used as is,
        and the audio tokens are generated from it. Otherwise, the ground truth text token
        is given to the model to generate audio tokens.

        Args:
            last_tokens (torch.LongTensor):
                The last tokens to generate from.
                shape is [B, K, T], where B is the batch size, K is the number of codebooks, and T is the number of tokens.
                T > 1 means it is the first step of generation when prompt tokens are given.
            next_text_tokens (torch.LongTensor):
                The next text tokens. shape is [B]
            force_text_token (torch.BoolTensor):
                If True, the ground truth text token is always used to generate audio tokens. shape is [B]
        Returns:
            Tuple of two torch.LongTensor:
            - sampled_tokens (torch.LongTensor):
                The sampled tokens. shape is [B, K, 1]
            - is_non_padding (torch.BoolTensor):
                The boolean tensor indicating whether the generated text token is not padding token. shape is [B]
        """
        if not self.moshi_lm.transformer.is_streaming:
            raise RuntimeError("moshi_lm.transformer should be in streaming mode")

        B, K, T = last_tokens.shape
        assert K == self.moshi_lm.num_codebooks, (
            f"Expected {self.moshi_lm.num_codebooks}, but got {K}"
        )

        # 1 tempral transformer forward
        if T == 1 and self.tempformer_forward_graphed is not None:
            temp_out, text_logits = self.tempformer_forward_graphed(last_tokens)
        else:
            # cannot forward multiple tokens to the graphed function
            temp_out, text_logits = self.moshi_lm.forward_text(last_tokens)
        assert temp_out.shape == (B, T, self.moshi_lm.dim), (
            f"Expected shape {(B, T, self.moshi_lm.dim)}, but got {temp_out.shape}"
        )
        assert text_logits.shape == (B, 1, T, self.moshi_lm.text_card), (
            f"Expected shape {(B, 1, T, self.moshi_lm.text_card)}, but got {text_logits.shape}"
        )

        temp_out = temp_out[:, -1][:, None]  # shape is [B, 1, dim]
        text_logits = text_logits[:, :, -1]  # shape is [B, 1, text_card]

        # 2 sample next text tokens
        text_token = sample_token(text_logits.float(), **self.text_sampling_params)
        assert text_token.shape == (B, 1), f"Expected shape {(B, 1)}, but got {text_token.shape}"
        text_token = text_token[:, 0]  # shape is [B]
        # print(f"{text_token=}")

        # 2.1 if text_token is not padding token, use ground truth text token
        is_non_padding = (text_token != self.moshi_lm.text_padding_token_id) & (
            text_token != self.moshi_lm.end_of_text_padding_id
        )
        next_text_forced = is_non_padding | force_text_token
        text_token = torch.where(next_text_forced, next_text_tokens, text_token)
        # print(f"{text_token=}")

        # 3 sample next audio tokens
        audio_tokens = self.depformer_step_graphed(text_token, temp_out)
        assert audio_tokens.shape == (B, self.moshi_lm.dep_q), audio_tokens.shape

        sampled_tokens = torch.cat([text_token[:, None], audio_tokens], dim=-1)
        assert sampled_tokens.shape == (B, self.moshi_lm.num_codebooks), sampled_tokens.shape
        sampled_tokens = sampled_tokens[..., None]  # shape is [B, num_codebooks, 1]

        return sampled_tokens, is_non_padding

    def generate(
        self,
        list_of_text_tokens: list[list[int]],
        generation_length: int,
        text_generation_params: dict[str, any],
        audio_generation_params: dict[str, any],
        prompt_tokens: torch.LongTensor,
        num_speaker_embedding_frames: int = 0,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """
        Generate audio streams from the given text tokens.

        Args:
            list_of_text_tokens (list[list[int]]):
                The text tokens to guide the audio generation.
                shape is [B, Tt], where B is the batch size and Tt is the number of text tokens.
            generation_length (int):
                The number of tokens to generate.
            generation_params (dict[str, any]):
                The generation parameters.
            prompt_tokens (torch.LongTensor):
                The prompt tokens to guide the audio generation.
                shape is [B, K, Tp], where B is the batch size, K is the number of codebooks, and Tp is the number of prompt tokens.
            num_speaker_embedding_frames (int):
                The number of speaker embedding frames to add to the head of text tokens.
                Set to 0 if speaker embeddings are not used.
        """
        device = next(self.moshi_lm.parameters()).device
        tensor_factory = {"dtype": torch.long, "device": device}
        B = len(list_of_text_tokens)
        assert prompt_tokens.shape[0] == B, f"Batch size mismatch: {prompt_tokens.shape[0]} != {B}"

        # fill final padding tokens
        max_delay = max(self.moshi_lm.delays)
        list_of_text_tokens = [
            tt + [self.moshi_lm.text_padding_token_id] * max_delay for tt in list_of_text_tokens
        ]

        # make tensor of text tokens
        len_text_tokens = torch.tensor(
            [len(text_tokens) + max_delay for text_tokens in list_of_text_tokens], **tensor_factory
        )
        text_tokens = torch.full(
            (B, len_text_tokens.max().item() + 1), self.moshi_lm.zero_token_id, **tensor_factory
        )
        for i, tt in enumerate(list_of_text_tokens):
            text_tokens[i, : len(tt)] = torch.tensor(tt, **tensor_factory)

        # make initial tokens
        prompt_tokens = prompt_tokens.to(device)

        # prepare generation
        self.prepare_generation(B, text_generation_params, audio_generation_params)

        # generate
        list_of_tokens = []
        num_consumed_text_tokens = torch.zeros(B, **tensor_factory)  # [B]
        text_tokens_finished = num_consumed_text_tokens >= len_text_tokens  # [B]
        generation_finished = torch.zeros(B, dtype=torch.bool, device=device)  # [B]
        audio_delay_counter = torch.tensor(self.moshi_lm.delays[1:], **tensor_factory)  # [K-1]
        audio_delay_counter -= (
            prompt_tokens.shape[-1]  # prompt tokens
            - num_speaker_embedding_frames  # exclude speaker embedding frames
            - 1  # exclude the initial token
        )
        last_tokens = prompt_tokens
        for _step in range(generation_length):
            # get next text token based on the number of consumed text tokens
            next_text_tokens = text_tokens[torch.arange(B), num_consumed_text_tokens]  # [B]
            # force_text_token = torch.tensor([False]*B, dtype=torch.bool, device=device)
            force_text_token = next_text_tokens.eq(
                self.moshi_lm.text_padding_token_id
            )  # fill final padding tokens
            # generate
            last_tokens, is_non_padding = self.step(
                last_tokens=last_tokens,
                next_text_tokens=next_text_tokens,
                force_text_token=force_text_token,
            )
            num_consumed_text_tokens += (
                ~text_tokens_finished  # do not consume text token if generation is finished
                & (
                    force_text_token | is_non_padding
                )  # consume text token if it is not padding token
            ).long()
            text_tokens_finished = num_consumed_text_tokens >= len_text_tokens
            generation_finished |= last_tokens[:, 0, 0] == self.moshi_lm.zero_token_id

            # update audio delay counter
            audio_delay_counter -= 1
            # overwrite the last tokens with intial audio tokens if the audio stream is waiting delay
            waiting_audio_stream_indices = (audio_delay_counter >= 0).repeat(B, 1)
            last_tokens[:, 1:][waiting_audio_stream_indices] = self.moshi_lm.initial_token_id

            list_of_tokens.append(last_tokens)

            if generation_finished.all():
                break

        # end generation
        self.finish_generation()

        generated_tokens = torch.cat(list_of_tokens, dim=-1)  # [B, K, T]
        assert generated_tokens.shape == (B, self.moshi_lm.num_codebooks, _step + 1), (
            f"Expected shape {(B, self.moshi_lm.num_codebooks, _step + 1)}, but got {generated_tokens.shape}"
        )

        # get generated length
        generated_length = torch.zeros(B, **tensor_factory)
        for b in range(B):
            # find first zero token (-1)
            zero_indices = generated_tokens[b, 0].eq(self.moshi_lm.zero_token_id).nonzero()
            if zero_indices.numel() > 0:
                generated_length[b] = zero_indices[0].item()
            else:
                generated_length[b] = _step + 1

        return generated_tokens, generated_length