#!/usr/bin/env python
"""Run Moshi finetuning with an explicit timeout for DeepSpeed subgroups."""

from __future__ import annotations

import os
import random
import runpy
import sys
from datetime import timedelta
from pathlib import Path

import torch.distributed
import torch.utils.data


timeout_seconds = int(os.environ["MOSHI_NCCL_SUBGROUP_TIMEOUT_SEC"])
original_new_group = torch.distributed.new_group


def new_group_with_timeout(*args, **kwargs):
    # DeepSpeed 0.15 calls new_group(ranks) without a timeout. PyTorch then
    # assigns its 600-second NCCL default, independently of the world group.
    if "timeout" not in kwargs:
        kwargs["timeout"] = timedelta(seconds=timeout_seconds)
    return original_new_group(*args, **kwargs)


torch.distributed.new_group = new_group_with_timeout
print(f"MOSHI_NCCL_SUBGROUP_TIMEOUT_SEC={timeout_seconds}", flush=True)


bucket_size = int(os.environ.get("MOSHI_LENGTH_BUCKET_SIZE", "0"))
if bucket_size > 0:
    bucket_seed = int(os.environ.get("MOSHI_DATALOADER_SEED", "1"))
    original_data_loader = torch.utils.data.DataLoader

    class LengthBucketSampler(torch.utils.data.Sampler):
        """Yield every example once, randomly ordered within shuffled length buckets."""

        def __init__(self, dataset, size: int, seed: int):
            lengths = dataset["num_frames"]
            ordered = sorted(range(len(lengths)), key=lengths.__getitem__)
            buckets = [ordered[i : i + size] for i in range(0, len(ordered), size)]
            rng = random.Random(seed)
            for bucket in buckets:
                rng.shuffle(bucket)
            rng.shuffle(buckets)
            self.indices = [index for bucket in buckets for index in bucket]
            print(
                f"MOSHI_LENGTH_BUCKET_SIZE={size} seed={seed} "
                f"examples={len(self.indices)} buckets={len(buckets)}",
                flush=True,
            )

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    class LengthBucketDataLoader(original_data_loader):
        def __init__(self, dataset, *args, **kwargs):
            if kwargs.get("shuffle", False) and "num_frames" in dataset.column_names:
                kwargs["shuffle"] = False
                kwargs["sampler"] = LengthBucketSampler(dataset, bucket_size, bucket_seed)
            super().__init__(dataset, *args, **kwargs)

    torch.utils.data.DataLoader = LengthBucketDataLoader

finetune_path = Path.cwd() / "finetune.py"
sys.path.insert(0, str(finetune_path.parent))
runpy.run_path(str(finetune_path), run_name="__main__")
