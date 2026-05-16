"""Convert a mono-trained MoshiForFinetuning into the mstts (multi-stream
TTS) format by extending the depth transformer's modules for the user
stream, then re-saving with updated kwargs (dep_q 8 -> 16,
depformer_context 8 -> 16).

This is the Stage 1 -> Stage 2 bridge of the laboro-free mstts rebuild:
  Stage 1 (mono) output (single-stream Japanese speech) -> extended -> input
  to mstts Stage 2 (J-CHAT multi-stream training).
"""

import argparse
from copy import deepcopy

import torch

from models import (
    AutoMoshiForFinetuning,
    extend_moshi_modules_for_user_stream,
)


def main(args: argparse.Namespace) -> None:
    print(f"Loading mono model from {args.mono_dir}")
    # `from_pretrained` runs `expose_linear_weights_for_zero3` internally, so
    # `mono` already has bare-parameter form (linear_in_weight, out_proj_weight).
    mono = AutoMoshiForFinetuning.from_pretrained(
        save_dir=args.mono_dir,
        device=torch.device("cpu"),
        dtype=torch.float32,  # extend in fp32, downcast on save
    )

    print("Extending depth transformer modules for user stream ...")
    # Returns a deepcopy with doubled depformer modules; still in exposed form.
    extended = extend_moshi_modules_for_user_stream(mono)

    # Update kwargs to reflect doubled dep_q (8 -> 16) and matching context.
    # NB: we do NOT call `from_original_moshi_lm` because that would re-run
    # expose_linear_weights_for_zero3 on an already-exposed model and crash.
    extended.moshi_lm_kwargs = deepcopy(mono.moshi_lm_kwargs)
    extended.moshi_lm_kwargs["dep_q"] = 16  # 8 (moshi) + 8 (user)
    extended.moshi_lm_kwargs["depformer_context"] = 16

    extended = extended.to(getattr(torch, args.model_dtype))

    print(f"Saving extended mstts init to {args.save_dir}")
    extended.save_pretrained(args.save_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mono_dir", type=str, required=True, help="Path to mono fp32 checkpoint dir."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Output dir for the extended mstts init."
    )
    parser.add_argument(
        "--model_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    main(parser.parse_args())
