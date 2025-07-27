import argparse
from copy import deepcopy

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from moshi.models import loaders

from models import (
    MoshiForFinetuning,
    extend_moshi_modules_for_user_stream,
)


def init_embedding_module(
    emb: nn.Embedding,
    retain_token_ids: list[int],
) -> nn.Embedding:
    # Initialize the embedding module with a Gaussian distribution
    dtype = emb.weight.dtype
    emb_weights = emb.weight.data
    mean = emb_weights.mean(dim=0)
    vocab_size = emb_weights.size()[0]
    sigma = ((emb_weights - mean).T @ (emb_weights - mean)) / vocab_size
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        mean.to(torch.float32),
        covariance_matrix=1e-5 * sigma.to(torch.float32),
    )  # MultivariateNormal is supported only for float32

    new_emb_weights = torch.stack(
        tuple(dist.sample() for _ in range(vocab_size)),
        dim=0,
    ).to(dtype)
    for token_id in retain_token_ids:
        if token_id >= vocab_size:
            raise ValueError(f"Token id {token_id} is out of range of the vocab_size {vocab_size}")
        new_emb_weights[token_id] = emb_weights[token_id]
    emb.weight.data = new_emb_weights
    return emb


def main(args):
    moshi_lm_kwargs = deepcopy(loaders._lm_kwargs)
    moshi_lm = loaders.get_moshi_lm(
        hf_hub_download(args.moshi_lm_repo, args.moshi_lm_name),
        device="cpu",
    )
    if args.init_text_embeddings:
        print("Initializing the text embedding modules...")
        if args.retain_text_token_ids:
            print(f"Reusing the embeddings of the text tokens: {args.retain_text_token_ids}")
        init_embedding_module(moshi_lm.text_emb, args.retain_text_token_ids)
        init_embedding_module(moshi_lm.depformer_text_emb, args.retain_text_token_ids)

    if args.extend_modules_for_user_stream:
        print("Extending the depth transformer's modules for user stream...")
        moshi_lm_kwargs.update(
            {
                "dep_q": 16,  # 8(moshi) + 8(user)
                "depformer_context": 16,  # 8(moshi) + 8(user)
            }
        )
        moshi_lm = extend_moshi_modules_for_user_stream(moshi_lm)

    print("Converting the original MoshiLM to MoshiForFinetuning...")
    moshi_lm = MoshiForFinetuning.from_original_moshi_lm(
        moshi_lm=moshi_lm, moshi_lm_kwargs=moshi_lm_kwargs
    )
    moshi_lm = moshi_lm.to(getattr(torch, args.model_dtype))

    print(f"Saving the initialized model to {args.save_dir}...")
    moshi_lm.save_pretrained(args.save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory path to save the initialized model",
    )
    parser.add_argument(
        "--moshi_lm_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Repository name of the Moshi model",
    )
    parser.add_argument(
        "--moshi_lm_name",
        type=str,
        default="model.safetensors",
        help="Model name of the Moshi model",
    )
    parser.add_argument(
        "--model_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Data type of the model",
    )
    parser.add_argument(
        "--init_text_embeddings",
        action="store_true",
        help=(
            "Initialize the text embedding modules. Use this flag if you "
            "want to change the text tokenizer"
        ),
    )
    parser.add_argument(
        "--retain_text_token_ids",
        nargs="+",
        type=int,
        default=[0, 3, 32000],
        help="List of text token ids that reuse their original embeddings",
    )
    parser.add_argument(
        "--extend_modules_for_user_stream",
        action="store_true",
        help="Extend the depth transformer's modules to model user stream",
    )
    args = parser.parse_args()
    main(args)
