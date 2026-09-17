"""Bounded CPU memory cache with optional persistent topology files."""
from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import torch

from .config import CoarseningConfig
from .topology import CoarseTopology, TOPOLOGY_VERSION, build_topology, build_v2_topology, topology_fingerprint


class TopologyCache:
    """Reuse discrete topology only; returned objects must be treated as read-only.

    Each worker/process owns its memory cache. Atomic disk replacement allows
    workers to share a directory (simultaneous cold misses may compute twice).
    No neural activations, weights or autograd graphs are stored.
    """

    def __init__(self, directory: str | Path | None = None, *, max_memory_entries: int = 128):
        if not isinstance(max_memory_entries, int) or max_memory_entries < 0:
            raise ValueError("max_memory_entries must be a nonnegative integer")
        self.directory = Path(directory) if directory is not None else None
        self.max_memory_entries = max_memory_entries
        self._memory: OrderedDict[str, CoarseTopology] = OrderedDict()
        self.hits = 0
        self.disk_hits = 0
        self.misses = 0

    @property
    def stats(self) -> dict[str, int]:
        return dict(hits=self.hits, disk_hits=self.disk_hits, misses=self.misses,
                    memory_entries=len(self._memory))

    def clear_memory(self) -> None:
        self._memory.clear()

    def _remember(self, key, topology):
        if self.max_memory_entries:
            self._memory[key] = topology
            self._memory.move_to_end(key)
            while len(self._memory) > self.max_memory_entries:
                self._memory.popitem(last=False)

    def get_or_build(self, num_nodes, edge_index, config=None, *, node_ids=None,
                     node_labels=None, edge_labels=None) -> CoarseTopology:
        config = config or CoarseningConfig()
        labels = dict(node_ids=node_ids, node_labels=node_labels, edge_labels=edge_labels)
        key = topology_fingerprint(num_nodes, edge_index, config, **labels)
        if key in self._memory:
            self.hits += 1
            self._memory.move_to_end(key)
            return self._memory[key]
        path = self.directory / f"{key}.pt" if self.directory is not None else None
        if path is not None and path.exists():
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload["version"] != TOPOLOGY_VERSION or payload["key"] != key:
                    raise ValueError("Cache metadata mismatch")
                topology = CoarseTopology(**payload["topology"])
                if topology.input_fingerprint != key:
                    raise ValueError("Cache input fingerprint mismatch")
            except Exception as exc:
                raise RuntimeError(f"Cannot read topology cache {path}; remove this file and precompute again") from exc
            self.disk_hits += 1
        else:
            builder = build_v2_topology if getattr(config, "algorithm", None) == "v2_exclusive_regions" else build_topology
            topology = builder(num_nodes, edge_index, config, **labels)
            self.misses += 1
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Save only tensors, lists, scalars and dictionaries, no custom pickle class.
                payload = dict(version=TOPOLOGY_VERSION, key=key, topology=asdict(topology))
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
                        temporary = Path(handle.name)
                        torch.save(payload, handle)
                    os.replace(temporary, path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        self._remember(key, topology)
        return topology
