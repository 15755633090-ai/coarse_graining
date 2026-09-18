"""Deterministic sortish batching that reduces padded-atom waste."""
from __future__ import annotations

import math

import torch
from torch.utils.data import Sampler


class AtomCountBucketBatchSampler(Sampler[list[int]]):
    """Shuffle locally, sort within large pools, then shuffle whole batches."""

    def __init__(
        self, atom_counts, batch_size: int, *, shuffle: bool = True,
        drop_last: bool = False, seed: int = 0, bucket_size_multiplier: int = 20,
    ):
        self.atom_counts = torch.as_tensor(atom_counts, dtype=torch.long).cpu()
        if self.atom_counts.ndim != 1 or self.atom_counts.numel() == 0:
            raise ValueError("atom_counts must be a nonempty one-dimensional sequence")
        if bool((self.atom_counts < 1).any()):
            raise ValueError("atom counts must be positive")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not isinstance(bucket_size_multiplier, int) or bucket_size_multiplier < 1:
            raise ValueError("bucket_size_multiplier must be positive")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_size_multiplier = bucket_size_multiplier
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.atom_counts.numel() // self.batch_size
        return math.ceil(self.atom_counts.numel() / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        count = self.atom_counts.numel()
        order = torch.randperm(count, generator=generator) if self.shuffle else torch.arange(count)
        pool_size = self.batch_size * self.bucket_size_multiplier
        batches = []
        for start in range(0, count, pool_size):
            pool = order[start:start + pool_size]
            stable = torch.argsort(self.atom_counts[pool], stable=True)
            pool = pool[stable]
            for offset in range(0, pool.numel(), self.batch_size):
                batch = pool[offset:offset + self.batch_size]
                if batch.numel() == self.batch_size or not self.drop_last:
                    batches.append(batch.tolist())
        if self.shuffle and len(batches) > 1:
            batch_order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in batch_order]
        yield from batches
