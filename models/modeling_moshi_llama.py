from dataclasses import dataclass
from functools import partial
from typing import Optional, Union, List
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    DynamicCache,
    LlamaConfig,
    LlamaModel,
)

from moshi.models.lm import (
    LMModel,
    ScaledEmbedding,
    StreamingContainer,
    StreamingTransformer,
    StreamingModule,
)


@dataclass
class _LlamaState:
    """
    Streaming state for Llama model.
    """

    past_key_values: DynamicCache

    def reset(self) -> None:
        """
        Reset the state.
        """
        self.past_key_values = DynamicCache()


class LlamaForMoshiLM(LlamaModel, StreamingModule[_LlamaState]):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        del self.embed_tokens  # we will use our own embeddings

    def get_input_embeddings(self):
        raise NotImplementedError("Use the embeddings in the MoshiLMwithLlama model.")

    def set_input_embeddings(self, value):
        raise NotImplementedError("Use the embeddings in the MoshiLMwithLlama model.")

    def _init_streaming_state(self, batch_size: int) -> _LlamaState:
        return _LlamaState(past_key_values=DynamicCache())

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.FloatTensor:
        state = self._streaming_state

        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if state is not None:
            model_inputs.update(
                {
                    "use_cache": True,
                    "past_key_values": state.past_key_values,
                }
            )

        output = super().forward(**model_inputs)

        if state is not None:
            # Update the state with the new past key values
            state.past_key_values = output.past_key_values

        return output.last_hidden_state  # Return the hidden states of the last layer


class MoshiLlama(LMModel):
    def __init__(
        self,
        llama_name_or_path: str,
        delays: List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: Optional[int] = None,
        end_of_text_padding_id: Optional[int] = None,
        context: Optional[int] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super(StreamingContainer, self).__init__()

        llama_config = LlamaConfig.from_pretrained(llama_name_or_path)
        # check consistency with keyword arguments and config
        assert (
            text_card == llama_config.vocab_size
        ), f"Text card ({text_card}) must match the LlamaConfig vocab size ({llama_config.vocab_size})."
        assert (
            dim == llama_config.hidden_size
        ), f"Dimension ({dim}) must match the LlamaConfig hidden size ({llama_config.hidden_size})."
        assert (
            num_heads == llama_config.num_attention_heads
        ), f"Number of heads ({num_heads}) must match the LlamaConfig num_attention_heads ({llama_config.num_attention_heads})."

        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        assert (
            len(delays) == self.num_codebooks
        ), f"Expected {self.num_codebooks} delays, got {len(delays)}."
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self._end_of_text_padding_id = end_of_text_padding_id
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        self.emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, self.dim) for _ in range(n_q)]
        )
        # Text card + padding token (if not in the original tokenizer)
        extra_text = self.existing_text_padding_id is None
        # Unlike for audio, here we authorize the model to output the special token.
        self.text_emb = EmbeddingFactory(self.text_card + 1, self.dim)
        self.text_linear = nn.Linear(
            self.dim, self.text_card + extra_text, bias=bias_proj
        )
        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        self.transformer = LlamaForMoshiLM(llama_config)
        self.out_norm = nn.Identity()  # out_norm is defined in the LlamaModel
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
            self.depformer_in = nn.ModuleList(
                [nn.Linear(self.dim, depformer_dim, bias=False) for _ in range(dep_q)]
            )
        else:
            self.depformer_in = nn.ModuleList(
                [nn.Linear(self.dim, depformer_dim, bias=False)]
            )
        # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
        self.depformer_emb = nn.ModuleList(
            [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )
        self.depformer_text_emb = EmbeddingFactory(self.text_card + 1, depformer_dim)
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        # Depformer follow its own cycle of streaming entirely contained in one time step
        # and should not follow the streaming of the steps dimensions.
        self.depformer.set_streaming_detached(True)
        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = nn.ModuleList(
            [nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        if self.existing_text_padding_id is None:
            return self.text_card
        else:
            return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        if self._end_of_text_padding_id is None:
            return 0
        else:
            return self._end_of_text_padding_id
