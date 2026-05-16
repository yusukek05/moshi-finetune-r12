import argparse
import collections
import json
import logging
import os
from datetime import timedelta
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, InitProcessGroupKwargs, set_seed
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from moshi.models.lm import LMModel
from sentencepiece import SentencePieceProcessor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data_utils import (
    AlternatingDatasetSampler,
    DataCollator,
    DataCollatorWithTextBatch,
    TextBatch,
    preprocess_function_for_singlechannel,
)
from finetune import (
    forward,
    get_parameters,
    postprocess_args,
    setup_argparser,
)
from models import AutoMoshiForFinetuning

logger = get_logger(__name__)


def setup_mono_argparser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--text_data_files",
        type=str,
        default=None,
        help="File patterns of the text data. Each file contains a list of json objects with 'text' field.",
    )
    parser.add_argument(
        "--text_tokenizer_file",
        type=str,
        default=None,
        help="Path to the tokenizer file for the text data. If not specified, will use a default tokenizer.",
    )
    parser.add_argument(
        "--text_tokenizer_repo",
        type=str,
        default="rinna/japanese-gpt2-medium",
        help="The repository of the tokenizer for the text data.",
    )
    parser.add_argument(
        "--text_tokenizer_name",
        type=str,
        default="spiece.model",
        help="The name of the tokenizer for the text data.",
    )
    parser.add_argument(
        "--text_batch_interval", type=int, default=4, help="The ratio of text batch to audio batch."
    )
    parser.add_argument(
        "--max_audio_delay",
        type=int,
        default=10,
        help="Range of audio delay to randomly sample from.",
    )
    parser.add_argument(
        "--use_kana",
        action="store_true",
        help="Whether the dataset has an A_kana stream to train on.",
    )


def check_mono_args(args):
    if args.moshi_speakers != ["A"]:
        raise ValueError("Only single speaker is supported for single channel finetuning.")

    if args.model_user_stream:
        raise ValueError("User stream is not supported for single channel finetuning.")

    if args.max_length is None:
        raise ValueError("max_length must be specified.")

    if args.min_length is not None:
        logger.warning("min_length is not used for single channel finetuning.")

    if not isinstance(args.seed, int):
        raise ValueError("seed must be specified.")


def text_forward(
    moshi_lm: LMModel, batch: TextBatch
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    assert batch.input_ids.dim() == 2 and batch.labels.dim() == 2, (
        "Text only batch should have shape (batch_size, seq_len)"
    )
    # Encode text
    text_emb = moshi_lm.text_emb(batch.input_ids)

    # Forward pass
    tempformer_out = moshi_lm.transformer(text_emb, attention_mask=batch.text_attention_mask)
    if moshi_lm.out_norm:
        tempformer_out = moshi_lm.out_norm(tempformer_out)
    text_logits = moshi_lm.text_linear(tempformer_out)

    # Compute loss
    text_logits = text_logits.float()
    text_logits = text_logits[:, :-1].contiguous()
    text_labels = batch.labels[:, 1:].contiguous()
    text_losses = F.cross_entropy(
        input=text_logits.view(-1, moshi_lm.text_card),
        target=text_labels.view(-1),
        ignore_index=moshi_lm.zero_token_id,
        reduction="none",
    ).view(text_labels.size())
    assert text_labels.shape == text_losses.shape, f"{text_labels.shape} != {text_losses.shape}"

    # Metrics
    text_accuracy = (text_logits.argmax(-1) == text_labels).float()

    non_pad_indices = (
        (text_labels != moshi_lm.text_padding_token_id)
        & (text_labels != moshi_lm.zero_token_id)  # zero token is ignored
    )

    loss = text_losses[non_pad_indices].mean()
    accuracy = text_accuracy[non_pad_indices].mean()
    log = {
        "loss/text_batch": loss.detach(),
        "accuracy/text_batch": accuracy.detach(),
    }
    return loss, log


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Finetune Moshi model on a custom dataset")
    setup_argparser(parser)
    setup_mono_argparser(parser)
    args = parser.parse_args()
    postprocess_args(args)
    check_mono_args(args)

    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "kwargs_handlers": [InitProcessGroupKwargs(timeout=timedelta(seconds=3600))],
    }

    if args.with_tracking:
        accelerator_kwargs.update(
            {
                "log_with": args.report_to,
                "project_dir": args.output_dir,
            }
        )

    if args.use_deepspeed and args.launcher != "accelerate":
        from accelerate import DeepSpeedPlugin

        accelerator_kwargs.update(
            {
                "deepspeed_plugin": DeepSpeedPlugin(
                    hf_ds_config=args.deepspeed_config_file,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                )
            }
        )

    accelerator = Accelerator(**accelerator_kwargs)
    args.num_processes = accelerator.num_processes

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
    logger.info(f"Loading Moshi model from {args.model_dir}")
    moshi_lm = AutoMoshiForFinetuning.from_pretrained(
        save_dir=args.model_dir,
        device=torch.device("cpu"),
        dtype=getattr(torch, args.model_dtype),
    )

    # Set activation checkpointing
    if args.activation_checkpointing:
        if args.use_deepspeed:
            assert os.environ.get("NO_TORCH_COMPILE"), "Not compatible with torch.compile"

            import deepspeed

            activation_checkpointing_kwargs = accelerator.deepspeed_plugin.hf_ds_config.config.get(
                "activation_checkpointing", {}
            )
            deepspeed.checkpointing.configure(
                mpu_=None, deepspeed_config=None, **activation_checkpointing_kwargs
            )
            moshi_lm.enable_activation_checkpointing(deepspeed.checkpointing.checkpoint)
        else:
            raise NotImplementedError(
                "Activation checkpointing is only supported with DeepSpeed for now."
            )

    # Set active/frozen for the finetuning
    for param in moshi_lm.parameters():
        param.requires_grad = False
    for param in get_parameters(moshi_lm, args.parameters_to_finetune):
        param.requires_grad = True
    for name, param in moshi_lm.named_parameters():
        logger.info(f"{name}: {param.requires_grad}")

    # Load the dataset
    with accelerator.main_process_first():
        logger.info(f"Loading train dataset from {args.train_data_files}")
        train_dataset = load_dataset(
            "parquet",
            split="train",
            data_files={"train": args.train_data_files},
            cache_dir=args.dataset_cache_dir,
        )
    if args.eval_data_files is not None:
        with accelerator.main_process_first():
            logger.info(f"Loading eval dataset from {args.eval_data_files}")
            eval_dataset = load_dataset(
                "parquet",
                split="validation",
                data_files={"validation": args.eval_data_files},
                cache_dir=args.dataset_cache_dir,
            )
    else:
        eval_dataset = None

    # Preprocess the dataset
    preprocessing_kwargs = {
        "max_length": args.max_length,
        "num_audio_codebooks": moshi_lm.num_audio_codebooks,
        "max_audio_delay": args.max_audio_delay,
        "initial_token_ids": [moshi_lm.text_initial_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "padding_token_ids": [moshi_lm.text_padding_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "zero_token_id": moshi_lm.zero_token_id,
        "use_kana": args.use_kana,
    }
    with accelerator.main_process_first():
        # only main process preprocesses the dataset, then others will use the resulted cache
        train_dataset = train_dataset.map(
            preprocess_function_for_singlechannel,
            remove_columns=train_dataset.column_names,
            batched=True,
            num_proc=args.dataset_processing_workers,
            fn_kwargs=preprocessing_kwargs,
            desc="Preprocessing train dataset",
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_function_for_singlechannel,
                remove_columns=eval_dataset.column_names,
                batched=True,
                num_proc=args.dataset_processing_workers,
                fn_kwargs=preprocessing_kwargs,
                desc="Preprocessing validation dataset",
            )

    # Preprocess the text dataset
    if args.text_data_files is not None:
        text_tokenizer_file = args.text_tokenizer_file
        if text_tokenizer_file is None:
            text_tokenizer_file = hf_hub_download(
                args.text_tokenizer_repo, args.text_tokenizer_name
            )
        logger.info(f"Loading text tokenizer from {text_tokenizer_file}")
        text_tokenizer = SentencePieceProcessor(text_tokenizer_file)
        with accelerator.main_process_first():
            logger.info(f"Loading text dataset from {args.text_data_files}")
            text_dataset = load_dataset(
                "json",
                split="train",
                data_files={"train": args.text_data_files},
                cache_dir=args.dataset_cache_dir,
            )

        bos_id = moshi_lm.end_of_text_padding_id

        def tokenize_text_func(batched_examples: dict[str, list[str]]) -> dict[str, list[int]]:
            """
            Tokenize the text data.
            """
            text_ids = text_tokenizer.encode(
                [t.lower() for t in batched_examples["text"]],
            )
            text_ids = [[bos_id] + ids for ids in text_ids]
            return {"text_stream": text_ids}

        with accelerator.main_process_first():
            text_dataset = text_dataset.map(
                tokenize_text_func,
                batched=True,
                num_proc=16,
                remove_columns=text_dataset.column_names,
                desc="Tokenizing text data",
            )

        max_length = args.max_length

        def group_text(batched_examples: dict[str, list[int]]) -> dict[str, list[int]]:
            """
            Make blocks of text data.
            """
            text_ids = list(chain(*batched_examples["text_stream"]))
            text_blocks = []
            num_blocks = -(-len(text_ids) // max_length)  # ceil division
            text_blocks = [block.tolist() for block in np.array_split(text_ids, num_blocks)]
            return {"text_stream": text_blocks}

        with accelerator.main_process_first():
            text_dataset = text_dataset.map(
                group_text,
                batched=True,
                num_proc=16,
                remove_columns=text_dataset.column_names,
                desc="Making blocks of text data",
            )

    else:
        text_dataset = None

    # Prepare dataloaders
    if text_dataset is None:
        data_collator = DataCollator(zero_token_id=moshi_lm.zero_token_id)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=True,
        )
        text_dataset_len = 0

    else:
        data_collator = DataCollatorWithTextBatch(zero_token_id=moshi_lm.zero_token_id)
        batch_sampler = AlternatingDatasetSampler(
            base_dataset_indices=list(range(len(train_dataset))),
            alt_dataset_indices=[i + len(train_dataset) for i in range(len(text_dataset))],
            base_ratio=args.text_batch_interval,
            batch_size=args.per_device_train_batch_size,
            num_processes=accelerator.num_processes,
            seed=args.seed,
        )
        train_dataset = concatenate_datasets([train_dataset, text_dataset])
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=data_collator,
        )
        # to inform accelerator of the batch size
        train_dataloader._DataLoader__initialized = False
        train_dataloader.batch_size = batch_sampler.batch_size
        train_dataloader._DataLoader__initialized = True

        text_dataset_len = len(text_dataset)

    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )
    else:
        eval_dataloader = None

    global_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    global_num_steps_per_epoch = len(train_dataloader)
    global_num_steps = args.num_train_epochs * global_num_steps_per_epoch
    local_num_steps_per_epoch = -(-global_num_steps_per_epoch // accelerator.num_processes)
    local_num_steps = args.num_train_epochs * local_num_steps_per_epoch

    # Prepare optimizer and learning rate scheduler
    param_groups = [
        {  # Temporal Transformer
            "params": get_parameters(moshi_lm, "tempformer"),
            "lr": args.tempformer_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {  # Depth Transformer
            "params": get_parameters(moshi_lm, "depformer"),
            "lr": args.depformer_learning_rate,
            "weight_decay": args.weight_decay,
        },
    ]
    if args.use_deepspeed:
        optimizer = DummyOptim(
            param_groups,
            lr=args.tempformer_learning_rate,  # for accelerator to set deepspeed's lr
            weight_decay=args.weight_decay,  # for accelerator to set deepspeed's weight decay
        )
        # `defaults["lr"]` is used by accelerator to set max_lr of deepspeed's scheduler
        # Ref: Accelerator._prepare_deepspeed()
        optimizer.defaults = {
            "lr": [args.tempformer_learning_rate, args.depformer_learning_rate],
        }

        lr_scheduler_type = None
        if "scheduler" in accelerator.deepspeed_plugin.hf_ds_config.config:
            lr_scheduler_type = accelerator.deepspeed_plugin.hf_ds_config.config["scheduler"][
                "type"
            ]
        if lr_scheduler_type is None:
            lr_scheduler = None
        elif lr_scheduler_type == "WarmupLR":
            lr_scheduler = DummyScheduler(
                optimizer=optimizer,
                warmup_num_steps=args.num_warmup_steps,
            )
        elif lr_scheduler_type == "WarmupDecayLR":
            lr_scheduler = DummyScheduler(
                optimizer=optimizer,
                warmup_num_steps=args.num_warmup_steps,
                total_num_steps=global_num_steps,
            )
        else:
            raise NotImplementedError(f"Unknown lr_scheduler_type: {lr_scheduler_type}")

    # Prepare everything with our `accelerator`.
    moshi_lm, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        moshi_lm, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Resume training
    current_steps = 0
    starting_epoch = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        current_steps = int(os.path.basename(args.resume_from_checkpoint).split("_")[1])
        starting_epoch = current_steps // local_num_steps_per_epoch

    # Initialize tracker
    config = vars(args)
    if args.use_deepspeed:
        config["deepspeed_config"] = json.load(open(args.deepspeed_config_file))
    if args.with_tracking and accelerator.is_main_process:
        wandb_init_kwargs = {
            "name": os.path.basename(args.output_dir),
        }
        if args.run_id_to_resume is not None:
            wandb_init_kwargs.update(
                {
                    "resume": "must",
                    "id": args.run_id_to_resume,
                }
            )
        accelerator.init_trackers(
            project_name=args.project_name,
            config=config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        config["run_id"] = accelerator.get_tracker(name="wandb", unwrap=True).id  # for later resume
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Start training
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)} (text: {text_dataset_len})")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {global_batch_size}"
    )
    logger.info(f"  Num total batches = {local_num_steps}")
    if args.resume_from_checkpoint:
        logger.info(f"  Resume from step {current_steps}")

    # Only show the progress bar once on each machine.
    pbar = tqdm(
        range(local_num_steps),
        initial=current_steps,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.resume_from_checkpoint and epoch == starting_epoch:
            # skip the first epoch if we resume from the middle
            # if accelerator.use_stateful_dataloader:
            #     active_dataloader = train_dataloader
            # else:
            num_batches_to_skip = (
                current_steps * args.gradient_accumulation_steps  # steps -> batches
            ) % local_num_steps_per_epoch
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, num_batches_to_skip
            )
        else:
            num_batches_to_skip = 0
            active_dataloader = train_dataloader

        logging_buffer = collections.defaultdict(list)

        for step, batch in enumerate(active_dataloader, start=num_batches_to_skip):
            moshi_lm.train()
            batch = batch.to(accelerator.device)

            # Forward pass
            if isinstance(batch, TextBatch):
                total_loss, log = text_forward(moshi_lm=moshi_lm, batch=batch)
            else:
                total_loss, log = forward(moshi_lm=moshi_lm, batch=batch, args=args)
            for key, value in log.items():
                logging_buffer[f"training_{key}"].append(value)
            # Backward pass
            accelerator.backward(total_loss)
            # The following optimization steps are handled by deepseed's backward() in accelerator,
            # so we don't need to do them here manually
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:
                pbar.update(1)
                current_steps += 1

                # Log metrics
                if current_steps % args.logging_steps == 0:
                    lrs = {
                        "tempformer": f"{optimizer.param_groups[0]['lr']:.3e}",
                        "depformer": f"{optimizer.param_groups[1]['lr']:.3e}",
                    }
                    log_str = (
                        f"Epoch: {epoch}, "
                        f"Steps: {current_steps}, "
                        f"LRs: {lrs}, "
                        f"Loss: {total_loss.item():.5f} "
                    )
                    if not isinstance(batch, TextBatch):
                        log_str += (
                            f"(text: {log['loss/text_total'].item():.5f}, "
                            f"audio: {log['loss/audio_total'].item():.5f})"
                        )
                    logger.info(log_str)
                    if args.with_tracking:
                        gathered_metrics = accelerator.gather(
                            {
                                key: torch.tensor(values, device=accelerator.device)
                                for key, values in logging_buffer.items()
                            }
                        )
                        accelerator.log(
                            {
                                **{
                                    key: values.nanmean()
                                    for key, values in gathered_metrics.items()
                                },
                                "learning_rate/tempformer": optimizer.param_groups[0]["lr"],
                                "learning_rate/depformer": optimizer.param_groups[1]["lr"],
                            },
                            step=current_steps,
                        )
                    logging_buffer = collections.defaultdict(list)  # reset

                # Evaluate the model
                if args.eval_steps is not None and current_steps % args.eval_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info(f"  Num examples = {len(eval_dataset)}")
                    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
                    logger.info(f"  Num steps = {len(eval_dataloader)}")
                    eval_logging_buffer = collections.defaultdict(list)
                    moshi_lm.eval()
                    for batch in tqdm(
                        eval_dataloader,
                        desc="Evaluating",
                        dynamic_ncols=True,
                        disable=not accelerator.is_main_process,
                    ):
                        batch = batch.to(accelerator.device)
                        with torch.no_grad():
                            if isinstance(batch, TextBatch):
                                _, log = text_forward(moshi_lm=moshi_lm, batch=batch)
                            else:
                                _, log = forward(moshi_lm=moshi_lm, batch=batch, args=args)
                            for key, value in log.items():
                                eval_logging_buffer[f"evaluation_{key}"].append(value)
                    if args.with_tracking:
                        gathered_metrics = accelerator.gather(
                            {
                                key: torch.tensor(values, device=accelerator.device)
                                for key, values in eval_logging_buffer.items()
                            }
                        )
                        accelerator.log(
                            {key: values.nanmean() for key, values in gathered_metrics.items()},
                            step=current_steps,
                        )

                # Save checkpoint
                if args.save_steps is not None and current_steps % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
                    accelerator.save_state(output_dir)

                # Stop training
                if args.max_train_steps is not None and current_steps >= args.max_train_steps:
                    break

    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
    accelerator.save_state(output_dir)


if __name__ == "__main__":
    main()
