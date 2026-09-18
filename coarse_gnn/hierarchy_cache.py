"""Bounded memory and atomic disk cache for hierarchy-only preprocessing."""
from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import torch

from .hierarchy import (
    HIERARCHY_VERSION,
    ComponentHierarchy,
    HierarchicalTopology,
    HierarchyConfig,
    HierarchyLevel,
    build_hierarchical_topology,
    hierarchy_input_fingerprint,
)


def _restore_topology(payload: dict) -> HierarchicalTopology:
    components = []
    for component in payload["components"]:
        component = dict(component)
        # Runtime memoization is intentionally not part of persistent identity.
        component.pop("_context_cache", None)
        levels = [HierarchyLevel(**level) for level in component["levels"]]
        components.append(ComponentHierarchy(
            atom_indices=component["atom_indices"],
            levels=levels,
            l1_distances=component["l1_distances"],
        ))
    return HierarchicalTopology(
        components=components,
        atom_order=payload["atom_order"],
        input_to_canonical=payload["input_to_canonical"],
        node_labels=payload["node_labels"],
        edges=payload["edges"],
        edge_labels=payload["edge_labels"],
        edge_type_keys=tuple(tuple(row) for row in payload["edge_type_keys"]),
        input_fingerprint=payload["input_fingerprint"],
        canonical_fingerprint=payload["canonical_fingerprint"],
        config=HierarchyConfig(**payload["config"]),
    )


class HierarchyCache:
    """Cache only discrete CPU topology; never cache activations or gradients."""

    def __init__(self, directory: str | Path | None = None, *, max_memory_entries: int = 128):
        if not isinstance(max_memory_entries, int) or max_memory_entries < 0:
            raise ValueError("max_memory_entries must be a nonnegative integer")
        self.directory = Path(directory) if directory is not None else None
        self.max_memory_entries = max_memory_entries
        self._memory: OrderedDict[str, HierarchicalTopology] = OrderedDict()
        self.hits = 0
        self.disk_hits = 0
        self.misses = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "disk_hits": self.disk_hits,
            "misses": self.misses,
            "memory_entries": len(self._memory),
        }

    def clear_memory(self) -> None:
        self._memory.clear()

    def _remember(self, key: str, topology: HierarchicalTopology) -> None:
        if not self.max_memory_entries:
            return
        self._memory[key] = topology
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)

    def get_or_build(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        config: HierarchyConfig | None = None,
        *,
        node_labels: torch.Tensor | None = None,
        edge_labels: torch.Tensor | None = None,
    ) -> HierarchicalTopology:
        config = config or HierarchyConfig()
        key = hierarchy_input_fingerprint(
            num_nodes, edge_index, config,
            node_labels=node_labels, edge_labels=edge_labels,
        )
        if key in self._memory:
            self.hits += 1
            self._memory.move_to_end(key)
            return self._memory[key]
        path = self.directory / f"{key}.pt" if self.directory is not None else None
        if path is not None and path.exists():
            try:
                saved = torch.load(path, map_location="cpu", weights_only=True)
                if saved["version"] != HIERARCHY_VERSION or saved["key"] != key:
                    raise ValueError("hierarchy cache metadata mismatch")
                topology = _restore_topology(saved["topology"])
                if topology.input_fingerprint != key:
                    raise ValueError("hierarchy cache input fingerprint mismatch")
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot read hierarchy cache {path}; remove it and precompute again"
                ) from exc
            self.disk_hits += 1
        else:
            topology = build_hierarchical_topology(
                num_nodes, edge_index, config,
                node_labels=node_labels, edge_labels=edge_labels,
            )
            self.misses += 1
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                saved = {
                    "version": HIERARCHY_VERSION,
                    "key": key,
                    "topology": asdict(topology),
                }
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=path.parent, suffix=".tmp", delete=False,
                    ) as handle:
                        temporary = Path(handle.name)
                        torch.save(saved, handle)
                    os.replace(temporary, path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        self._remember(key, topology)
        return topology

    def get_or_build_many(self, inputs) -> list[HierarchicalTopology]:
        """Prepare a batch without coupling cache identity to batch order."""
        return [
            self.get_or_build(
                item[0], item[1], item[2] if len(item) > 2 else None,
                node_labels=item[3] if len(item) > 3 else None,
                edge_labels=item[4] if len(item) > 4 else None,
            )
            for item in inputs
        ]
