import argparse
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM
from huggingface_hub import hf_hub_download
from moshi.models import loaders

from models import (
    MoshiLlama,
    MoshiLlamaForFinetuning,
    extend_moshi_modules_for_user_stream,
)


def extend_vocab_embedding(emb: nn.Embedding, new_vocab_size: int):
    dtype = emb.weight.dtype
    emb_weights = emb.weight.data
    mean = emb_weights.mean(dim=0)
    vocab_size = emb_weights.size()[0]
    if new_vocab_size <= vocab_size:
        raise ValueError(
            f"New vocab size {new_vocab_size} must be greater than current vocab size {vocab_size}"
        )

    sigma = ((emb_weights - mean).T @ (emb_weights - mean)) / vocab_size
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mean.to(torch.float32),
        covariance_matrix=1e-5 * sigma.to(torch.float32),
    )
    new_weights = torch.stack(
        tuple(dist.sample() for _ in range(new_vocab_size - vocab_size)), dim=0
    ).to(dtype)
    new_weights = torch.cat([emb_weights, new_weights], dim=0)
    new_emb = nn.Embedding(new_vocab_size, emb.embedding_dim, dtype=dtype)
    new_emb.weight.data = new_weights
    return new_emb


def main(args):
    """
    Modules to be initialized are:
    - (emb): random
    - (text_emb): initialized with `model.embed_tokens` in LlamaForCausalLM
    - (text_linear): initialized with `lm_head` in LlamaForCausalLM
    - (transformer): initialized with `model` LlamaForCausalLM
    - (out_norm): none
    - (depformer_in): random
    - (depformer_emb): optionally initialized with `depformer_emb` in Moshi
    - (depformer_text_emb): random
    - (depformer): optionally initialized with `depformer` in Moshi
    - (linears): optionally initialized with `linears` in Moshi
    """

    if args.moshi_llama_ft_dir:
        print(
            f"Loading the finetuned MoshiLlama model from {args.moshi_llama_ft_dir}..."
        )
        moshi_llama_ft = MoshiLlamaForFinetuning.from_pretrained(
            save_dir=args.moshi_llama_ft_dir,
            device=torch.device("cpu"),
            dtype=getattr(torch, args.model_dtype),
        )
        moshi_llama_kwargs = moshi_llama_ft.moshi_llama_kwargs
        moshi_llama = moshi_llama_ft.to_original_moshi_llama()
    else:
        print(
            f"Initializing a new MoshiLlama model from the repository {args.moshi_lm_repo}..."
        )
        # Initialize MoshiLlama model
        moshi_llama_kwargs = deepcopy(loaders._lm_kwargs)
        # Override the default MoshiLlama kwargs
        llama_config = LlamaConfig.from_pretrained(args.llama_repo)
        n_q = args.num_audio_codebooks
        dep_q = n_q
        delays = []
        delays += [0]  # text stream
        delays += [args.semantic_delay]  # moshi's semantic stream
        delays += [args.acoustic_delay] * (n_q - 1)  # moshi's acoustic streams
        if args.model_user_stream:
            n_q *= 2
            dep_q *= 2
            delays += [args.semantic_delay]  # user semantic stream
            delays += [args.acoustic_delay] * (args.num_audio_codebooks - 1)  # user acoustic streams
        moshi_llama_kwargs.update(
            {
                "n_q": n_q,
                "dep_q": dep_q,
                "dim": llama_config.hidden_size,
                "text_card": llama_config.vocab_size,
                "existing_text_padding_id": args.text_padding_token_id,
                "num_heads": llama_config.num_attention_heads,
                "context": llama_config.max_position_embeddings,
                "depformer_dim": args.depformer_dim,
                "depformer_num_heads": args.depformer_num_heads,
                "depformer_num_layers": args.depformer_num_layers,
                "depformer_context": dep_q,
                "delays": delays,
                "llama_name_or_path": args.llama_repo,
                "end_of_text_padding_id": args.end_of_text_padding_id,
            }
        )

        moshi_llama = MoshiLlama(
            device="cpu", dtype=getattr(torch, args.model_dtype), **moshi_llama_kwargs
        )

        # Load LlamaForCausalLM weights into MoshiLlama
        print(f"Applying the {args.llama_repo} weights to MoshiLlama...")
        # 1 text_emb
        llama = LlamaForCausalLM.from_pretrained(args.llama_repo)
        text_emb = extend_vocab_embedding(
            emb=llama.get_input_embeddings(),
            new_vocab_size=moshi_llama.text_emb.num_embeddings,
        )
        moshi_llama.text_emb.load_state_dict(text_emb.state_dict())
        # 2 transformer
        llama_sd = OrderedDict(
            {
                k: v
                for k, v in llama.model.state_dict().items()
                if k != "embed_tokens.weight"
            }
        )
        moshi_llama.transformer.load_state_dict(llama_sd)
        # 3 text_linear
        text_linear = llama.get_output_embeddings()
        moshi_llama.text_linear.load_state_dict(text_linear.state_dict())

        # Load Moshi's depformer weights into MoshiLlama
        if args.moshi_lm_repo is not None:
            assert (
                args.moshi_lm_file is not None
            ), "If `moshi_lm_repo` is provided, `moshi_lm_file` must also be specified."
            print(f"Applying the {args.moshi_lm_repo} weights to MoshiLlama...")
            moshi_lm = loaders.get_moshi_lm(
                hf_hub_download(args.moshi_lm_repo, args.moshi_lm_file),
                device="cpu",
            )
            # 4 depformer_emb
            moshi_llama.depformer_emb.load_state_dict(
                moshi_lm.depformer_emb.state_dict()
            )
            # 5 depformer
            moshi_llama.depformer.load_state_dict(moshi_lm.depformer.state_dict())
            # 6 linears
            moshi_llama.linears.load_state_dict(moshi_lm.linears.state_dict())
        else:
            print(
                "No Moshi model repository provided. Skipping weights initialization for depformer"
            )

    if args.duplicate_modules_for_user_stream:
        assert (
            not args.model_user_stream
        ), "User stream modules are already created in MoshiLlamaForFinetuning."
        print("Extending the depth transformer's modules for user stream...")
        delays = moshi_llama_kwargs["delays"].copy()
        delays += [args.semantic_delay]  # user semantic stream
        delays += [args.acoustic_delay] * (
            moshi_llama_kwargs["n_q"] - 1
        )  # user acoustic streams
        moshi_llama_kwargs.update(
            {
                "n_q": moshi_llama_kwargs["n_q"] * 2,  # 8(moshi) + 8(user)
                "dep_q": moshi_llama_kwargs["dep_q"] * 2,  # 8(moshi) + 8(user)
                "depformer_context": moshi_llama_kwargs["dep_q"]
                * 2,  # 8(moshi) + 8(user)
                "delays": delays,
            }
        )
        moshi_llama = extend_moshi_modules_for_user_stream(moshi_llama)

    # Convert to MoshiLlamaForFinetuning
    moshi_llama_ft = MoshiLlamaForFinetuning.from_original_moshi_llama(
        moshi_llama=moshi_llama, moshi_llama_kwargs=moshi_llama_kwargs
    )

    print(f"Saving the initialized model to {args.save_dir}...")
    moshi_llama_ft.save_pretrained(args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize a MoshiLlama model with a pretrained LlamaForCausalLM and Moshi."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory path to save the initialized model",
    )
    parser.add_argument(
        "--llama_repo",
        type=str,
        default=None,
        help="Hugging Face repository name of the pretrained LlamaForCausalLM model",
    )
    parser.add_argument(
        "--moshi_lm_repo",
        type=str,
        default=None,
        help="Hugging Face repository name of the pretrained Moshi model",
    )
    parser.add_argument(
        "--moshi_lm_file",
        type=str,
        default=None,
        help="Name of the Moshi model to load from the repository",
    )
    parser.add_argument(
        "--moshi_llama_ft_dir",
        type=str,
        default=None,
        help=(
            "Directory path to the finetuned MoshiLlama model. If provided, "
            "the model will be loaded from this directory instead of the "
            "Moshi repository."
        ),
    )
    parser.add_argument(
        "--model_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type of the model",
    )
    parser.add_argument(
        "--text_padding_token_id",
        type=int,
        default=3,
        help="Padding token ID for text embeddings",
    )
    parser.add_argument(
        "--end_of_text_padding_id",
        type=int,
        default=0,
        help="End of text padding token ID for text embeddings",
    )
    parser.add_argument(
        "--model_user_stream",
        action="store_true",
        help="Whether to create user stream modules in the MoshiLlama model",
    )
    parser.add_argument(
        "--duplicate_modules_for_user_stream",
        action="store_true",
        help=(
            "Duplicate modules for user stream in Depth Transformer. "
            "This is only applicable if `model_user_stream` is set to False."
        ),
    )

    parser.add_argument(
        "--num_audio_codebooks",
        type=int,
        default=8,
        help="Number of audio codebooks to use in the MoshiLlama model",
    )
    parser.add_argument(
        "--depformer_dim",
        type=int,
        default=1024,
        help="Hidden dimension of the depth transformer",
    )
    parser.add_argument(
        "--depformer_num_heads",
        type=int,
        default=16,
        help="Number of attention heads in the depth transformer",
    )
    parser.add_argument(
        "--depformer_num_layers",
        type=int,
        default=6,
        help="Number of layers in the depth transformer",
    )
    parser.add_argument(
        "--semantic_delay",
        type=int,
        default=0,
        help="Delay for semantic processing in the depth transformer",
    )
    parser.add_argument(
        "--acoustic_delay",
        type=int,
        default=1,
        help="Delay for acoustic processing in the depth transformer",
    )
    args = parser.parse_args()
    main(args)
