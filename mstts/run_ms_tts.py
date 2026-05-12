import os
import json
import logging
import argparse

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

import sentencepiece
from datasets import Dataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from huggingface_hub import hf_hub_download

from models import (
    AutoMoshiForFinetuning,
    MoshiForMultiStreamTTS
)

from data_utils import (
    delay_and_pad_streams,
    tokenize_text_chat_for_multistream_tts,
    undelay_tokens,
    find_last_non_padding,
)

logger = get_logger(__name__)

def set_mpi_env_vars():
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))
    world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))

    master_addr = os.environ.get("HOSTNAME", None)
    if master_addr is None:
        raise ValueError("HOSTNAME environment variable is not set")
    
    master_port = "29500"

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(world_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    torch.cuda.set_device(local_rank) # set default device

    return world_size, world_rank, local_rank


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for Multi-stream TTS model"
    )

    parser.add_argument(
        "--launcher",
        choices=["accelerate", "mpi"],
        required=True,
        help="Launcher type to use for distributed inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output"
    )
    parser.add_argument(
        "--text_chat_data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the text chat data"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the pre-trained Multi-stream TTS model (`model.safetensors`) "
            "and config for initializing model (`init_moshi_kwargs.json`)."
        )
    )
    parser.add_argument(
        "--model_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model data type",
    )
    parser.add_argument(
        "--text_tokenizer_file",
        type=str,
        default=None,
        help=(
            "Path to the text tokenizer file. If provided, it will override the `text_tokenizer_repo` and "
            "`text_tokenizer_name` arguments."
        ),
    )
    parser.add_argument(
        "--text_tokenizer_repo",
        type=str,
        default="rinna/japanese-gpt2-medium",
        help="Repository of the text tokenizer",
    )
    parser.add_argument(
        "--text_tokenizer_name",
        type=str,
        default="spiece.model",
        help="Name of the text tokenizer",
    )

    parser.add_argument(
        "--num_speaker_embedding_frames",
        type=int,
        default=0,
        help="Length of speaker embedding frames",
    )

    parser.add_argument(
        "--dataset_processing_workers",
        type=int,
        default=16,
        help="Number of workers to use for processing the dataset.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=".cache/huggingface/datasets",
        help="Directory to cache the datasets.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to process. If None, process all examples.",
    )

    # Generation parameters
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=1,
        help="Batch size for each device",
    )
    parser.add_argument(
        "--prompt_streams_path",
        type=str,
        default=None,
        help="Path to the prompt streams file (.npy). Data shape: (K, prompt_length).",
    )
    parser.add_argument(
        "--max_generation_length",
        type=int,
        default=2000,
        help="Maximum length of the generated sequences"
    )
    parser.add_argument(
        "--use_sampling",
        action="store_true",
        default=True,
        help="Use sampling for generation.",
    )
    parser.add_argument(
        "--text_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for text generation.",
    )
    parser.add_argument(
        "--audio_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for audio generation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k for sampling. Set to 0 for no top-k.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.,
        help="Top-p for sampling. Set to 0 for no top-p.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )

    args = parser.parse_args()

    # sanity check
    assert args.num_speaker_embedding_frames == 0, (
        "Speaker embedding frames are not supported in inference"
    )

    # post process args
    if args.launcher == "mpi":
        set_mpi_env_vars()

    return args

def main():
    args = parse_args()
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the model
    logger.info(f"Loading Multi-stream TTS model from {args.model_dir}")
    moshi_lm = AutoMoshiForFinetuning.from_pretrained(
        save_dir=args.model_dir,
        device=accelerator.device,
        dtype=getattr(torch, args.model_dtype),
    )
    _ = moshi_lm.eval()

    moshi_mstts = MoshiForMultiStreamTTS(moshi_lm=moshi_lm)

    # Load the text tokenizer
    text_tokenizer_path = args.text_tokenizer_file
    if text_tokenizer_path is None:
        text_tokenizer_path = hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
    logger.info(f"Loading text tokenizer from {text_tokenizer_path}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(text_tokenizer_path)

    # Load the dataset
    logger.info(f"Loading dataset from {args.text_chat_data_dir}")
    data = []
    for file in tqdm(
            os.listdir(args.text_chat_data_dir), desc="Loading data",
            dynamic_ncols=True, disable=not accelerator.is_local_main_process
        ):
        path = os.path.join(args.text_chat_data_dir, file)
        data.append({
            "dialogue_name": os.path.splitext(file)[0],
            "text_chat": [text for _, text in json.load(open(path))]
        })
    text_chat_dataset = Dataset.from_list(data)

    # Prepare the dataset
    def preprocess_text_chat(batched_examples: dict[str, list[str]]) -> dict[str, list[list[int] | int]]:
        list_of_text_tokens = []
        list_of_text_length = []
        for text_chat in batched_examples["text_chat"]:
            text_tokens = tokenize_text_chat_for_multistream_tts(
                text_chat=text_chat,
                text_tokenizer=text_tokenizer,
                main_speaker_bos_id=1,
                other_speaker_bos_id=2,
                main_speaker_first=False
            )
            list_of_text_tokens.append(text_tokens)
            list_of_text_length.append(len(text_tokens))
        return {"text_tokens": list_of_text_tokens, "text_length": list_of_text_length}

    with accelerator.main_process_first():
        text_chat_dataset = text_chat_dataset.map(
            preprocess_text_chat,
            batched=True,
            num_proc=args.dataset_processing_workers,
        )
    
    if args.num_examples is None:
        args.num_examples = len(text_chat_dataset)

    global_batch_size = args.per_device_batch_size * accelerator.num_processes
    local_num_steps = -(-args.num_examples // global_batch_size)
    text_chat_dataset = text_chat_dataset.select(range(args.num_examples))

    # sort dataset by length
    text_chat_dataset = text_chat_dataset.sort("text_length")

    def data_collator(examples: list[dict[str, str | list[int] | int]]) -> dict[str, list[str | list[int]]]:
        batch = {
            "dialogue_name": [ex["dialogue_name"] for ex in examples],
            "text_tokens": [ex["text_tokens"] for ex in examples],
        }
        return batch

    # Prepare the dataloader
    dataloader = DataLoader(
        text_chat_dataset,
        batch_size=args.per_device_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataset_processing_workers,
    )
    dataloader = accelerator.prepare(dataloader)


    # Prepare the prompt streams
    prompt_streams = np.load(args.prompt_streams_path)
    assert prompt_streams.shape[0] == moshi_lm.num_codebooks, (
        f"Number of prompt streams ({prompt_streams.shape[0]}) does not match number "
        f"of codebooks ({moshi_lm.num_codebooks})"
    )
    prompt_tokens = delay_and_pad_streams(
        list_of_streams=[prompt_streams],
        delays=moshi_lm.delays,
        initial_token_ids=[
                moshi_lm.text_initial_token_id
            ] + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        padding_token_ids=[
            moshi_lm.text_padding_token_id
        ] + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
    )
    prompt_tokens = torch.tensor(
        prompt_tokens[0], device=accelerator.device, dtype=torch.long
    ).expand(args.per_device_batch_size, -1, -1) 

    # save config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Make the output directory
    output_dir = os.path.join(args.output_dir, "generated_tokens")
    os.makedirs(output_dir, exist_ok=True)

    # Generate
    logger.info("***** Running generation *****")
    logger.info(f"  Num examples = {len(text_chat_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    logger.info(f"  Total batch size = {args.per_device_batch_size * accelerator.num_processes}")
    logger.info(f"  Num steps = {local_num_steps}")
    logger.info(f"  Max generation length = {args.max_generation_length}")
    logger.info(f"  Use sampling = {args.use_sampling}")
    if args.use_sampling:
        logger.info(f"  Sampling temperature (text, audio) = {args.text_temperature}, {args.audio_temperature}")
        logger.info(f"  Sampling top-k = {args.top_k}")
        logger.info(f"  Sampling top-p = {args.top_p}")

    progress_bar = tqdm(
        range(local_num_steps),
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True
    )

    for step, batch in enumerate(dataloader):
        actual_bs = len(batch["text_tokens"])
        gen_tokens, gen_lens = moshi_mstts.generate(
            list_of_text_tokens=batch["text_tokens"],
            generation_length=args.max_generation_length,
            text_generation_params={
                "use_sampling": args.use_sampling,
                "temp": args.text_temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
            },
            audio_generation_params={
                "use_sampling": args.use_sampling,
                "temp": args.audio_temperature,
                "top_k": args.top_k,
                "top_p": args.top_p
            },
            prompt_tokens=prompt_tokens[:actual_bs],
            num_speaker_embedding_frames=args.num_speaker_embedding_frames,
        )
        undelayed_tokens = undelay_tokens(gen_tokens, moshi_lm.delays)
        for i in range(len(batch["dialogue_name"])):
            dialogue_name = batch["dialogue_name"][i]
            output_path = os.path.join(output_dir, f"{dialogue_name}.npy")
            gen_len = gen_lens[i].item() - max(moshi_lm.delays)
            tokens_i = undelayed_tokens[i, :, :gen_len]
            last_index = find_last_non_padding(tokens_i[0], buffer_len=25)
            np.save(output_path, tokens_i[:, :last_index].cpu().numpy())

        progress_bar.update(1)
    progress_bar.close()

if __name__ == "__main__":
    main()
