import os
import json
import math
import logging
import argparse
import collections

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import load_dataset   
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed


from models import AutoMoshiForFinetuning
from data_utils import (
    preprocess_function_for_multistream_tts,
    DataCollator,
)
from finetune import setup_argparser as setup_ft_argparser
from finetune import (
    postprocess_args,
    get_parameters,
    forward,
)

logger = get_logger(__name__)

def setup_ms_tts_argparser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--main_speaker_bos_id", type=int, default=1,
        help="Text token id of the main speaker"
    )
    parser.add_argument(
        "--other_speaker_bos_id", type=int, default=2,
        help="Text token id of the other speaker"
    )

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Finetune Moshi model on a custom dataset"
    )
    setup_ft_argparser(parser)
    setup_ms_tts_argparser(parser)
    args = parser.parse_args()
    postprocess_args(args)

    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    if args.with_tracking:
        accelerator_kwargs.update({
            "log_with": args.report_to,
            "project_dir": args.output_dir,
        })

    if args.use_deepspeed and args.launcher != "accelerate":
        from accelerate import DeepSpeedPlugin
        accelerator_kwargs.update({
            "deepspeed_plugin": DeepSpeedPlugin(
                hf_ds_config=args.deepspeed_config_file,
                gradient_accumulation_steps=args.gradient_accumulation_steps,    
            )
        })

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
            raise NotImplementedError("Activation checkpointing is only supported with DeepSpeed for now.")

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
            "parquet", split="train",
            data_files={"train": args.train_data_files},
            cache_dir=args.dataset_cache_dir,
        )
    if args.eval_data_files is not None:
        with accelerator.main_process_first():
            logger.info(f"Loading eval dataset from {args.eval_data_files}")
            eval_dataset = load_dataset(
                "parquet", split="validation",
                data_files={"validation": args.eval_data_files},
                cache_dir=args.dataset_cache_dir,
            )
    else:
        eval_dataset = None

    # Preprocess the dataset
    preprocessing_kwargs = {
        "speakers": args.moshi_speakers,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "delays": moshi_lm.delays,
        "initial_token_ids": [
            moshi_lm.text_initial_token_id
        ] + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "padding_token_ids": [
            moshi_lm.text_padding_token_id
        ] + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "end_of_text_padding_token_id": moshi_lm.end_of_text_padding_id,
        "main_speaker_bos_id": args.main_speaker_bos_id,
        "other_speaker_bos_id": args.other_speaker_bos_id,
        "zero_token_id": moshi_lm.zero_token_id,
    }
    with accelerator.main_process_first():
        # only main process preprocesses the dataset, then others will use the resulted cache
        train_dataset = train_dataset.map(
            preprocess_function_for_multistream_tts,
            remove_columns=train_dataset.column_names,
            batched=True, num_proc=args.dataset_processing_workers,
            fn_kwargs=preprocessing_kwargs,
            desc="Preprocessing train dataset",
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_function_for_multistream_tts,
                remove_columns=eval_dataset.column_names,
                batched=True, num_proc=args.dataset_processing_workers,
                fn_kwargs=preprocessing_kwargs,
                desc="Preprocessing validation dataset",
            )
    
    data_collator = DataCollator(zero_token_id=moshi_lm.zero_token_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )
    else:
        eval_dataloader = None

    global_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    local_num_steps_per_epoch = math.ceil(len(train_dataset) / global_batch_size)
    global_num_steps_per_epoch = local_num_steps_per_epoch * accelerator.num_processes
    local_num_steps = args.num_train_epochs * local_num_steps_per_epoch
    global_num_steps = args.num_train_epochs * global_num_steps_per_epoch

    # Prepare optimizer and learning rate scheduler
    param_groups = [
        { # Temporal Transformer
            "params": get_parameters(moshi_lm, "tempformer"),
            "lr": args.tempformer_learning_rate,
            "weight_decay": args.weight_decay,
        },
        { # Depth Transformer
            "params": get_parameters(moshi_lm, "depformer"),
            "lr": args.depformer_learning_rate,
            "weight_decay": args.weight_decay,
        }
    ]
    if args.use_deepspeed:
        optimizer = DummyOptim(
            param_groups,
            lr=args.tempformer_learning_rate, # for accelerator to set deepspeed's lr
            weight_decay=args.weight_decay, # for accelerator to set deepspeed's weight decay
        )
        # `defaults["lr"]` is used by accelerator to set max_lr of deepspeed's scheduler
        # Ref: Accelerator._prepare_deepspeed()
        optimizer.defaults = {
            "lr": [args.tempformer_learning_rate, args.depformer_learning_rate],
        }

        lr_scheduler_type = None
        if "scheduler" in accelerator.deepspeed_plugin.hf_ds_config.config:
            lr_scheduler_type = accelerator.deepspeed_plugin.hf_ds_config.config["scheduler"]["type"]
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
            wandb_init_kwargs.update({
                "resume": "must",
                "id": args.run_id_to_resume,
            })
        accelerator.init_trackers(
            project_name=args.project_name,
            config=config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        config["run_id"] = accelerator.get_tracker(name="wandb", unwrap=True).id # for later resume
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Start training
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {global_batch_size}")
    logger.info(f"  Total optimization steps = {local_num_steps}")
    if args.resume_from_checkpoint:
        logger.info(f"  Resume from step {current_steps}")

    # Only show the progress bar once on each machine.
    pbar = tqdm(
        range(local_num_steps),
        initial=current_steps,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True
    )

    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.resume_from_checkpoint and epoch == starting_epoch:
            # skip the first epoch if we resume from the middle
            # if accelerator.use_stateful_dataloader:
            #     active_dataloader = train_dataloader
            # else:
            num_batches_to_skip = (
                current_steps * args.gradient_accumulation_steps # steps -> batches
            ) % local_num_steps_per_epoch
            active_dataloader = accelerator.skip_first_batches(train_dataloader, num_batches_to_skip)
        else:
            num_batches_to_skip = 0
            active_dataloader = train_dataloader

        logging_buffer = collections.defaultdict(list)

        for step, batch in enumerate(active_dataloader, start=num_batches_to_skip):
            moshi_lm.train()
            batch = batch.to(accelerator.device)
            # Forward pass
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

            if (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                pbar.update(1)
                current_steps += 1

                # Log metrics
                if current_steps % args.logging_steps == 0:
                    lrs = {
                        "tempformer": f"{optimizer.param_groups[0]["lr"]:.3e}",
                        "depformer": f"{optimizer.param_groups[1]["lr"]:.3e}",
                    }
                    logger.info((
                        f"Epoch: {epoch}, "
                        f"Steps: {current_steps}, "
                        f"LRs: {lrs}, "
                        f"Loss: {total_loss.item():.5f} "
                        f"(text: {log['loss/text_total'].item():.5f}, "
                        f"audio: {log['loss/audio_total'].item():.5f})"
                    ))
                    if args.with_tracking:
                        gathered_metrics = accelerator.gather({
                            key: torch.tensor(values, device=accelerator.device) for key, values in logging_buffer.items()
                        })
                        accelerator.log(
                            {
                                **{key: values.nanmean() for key, values in gathered_metrics.items()},
                                "learning_rate/tempformer": optimizer.param_groups[0]["lr"],
                                "learning_rate/depformer": optimizer.param_groups[1]["lr"],
                            },
                            step=current_steps
                        )
                    logging_buffer = collections.defaultdict(list) # reset

                # Evaluate the model
                if args.eval_steps is not None and current_steps % args.eval_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info(f"  Num examples = {len(eval_dataset)}")
                    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
                    logger.info(f"  Num steps = {len(eval_dataloader)}")
                    eval_logging_buffer = collections.defaultdict(list)
                    moshi_lm.eval()
                    for batch in tqdm(
                        eval_dataloader, desc="Evaluating", dynamic_ncols=True,
                        disable=not accelerator.is_main_process
                    ):
                        batch = batch.to(accelerator.device)
                        with torch.no_grad():
                            _, log = forward(moshi_lm=moshi_lm, batch=batch, args=args)
                            for key, value in log.items():
                                eval_logging_buffer[f"evaluation_{key}"].append(value)
                    if args.with_tracking:
                        gathered_metrics = accelerator.gather({
                            key: torch.tensor(values, device=accelerator.device) for key, values in eval_logging_buffer.items()
                        })
                        accelerator.log(
                            {key: values.nanmean() for key, values in gathered_metrics.items()},
                            step=current_steps
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
