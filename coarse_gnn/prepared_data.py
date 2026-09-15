"""Cache the original dataset's graph objects and attach verified CPU topology."""
from __future__ import annotations

from .diffusion_adapter import precompute_batch_topologies
from .packed import PreparedGraphBatch


class PreparedPropertyDataset:
    def __init__(self, source, original_collate, cache, coarsening):
        self.source = source
        self.original_collate = original_collate
        self.cache = cache
        self.coarsening = coarsening
        self.rows = {}

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if index not in self.rows:
            row = self.source[index]  # Exact original featurizer and invalid-input policy.
            graph = self.original_collate([row]).graph
            topology = precompute_batch_topologies(graph, self.cache, self.coarsening)[0]
            self.rows[index] = (*row, topology)
        return self.rows[index]


def prepared_tools(legacy, cache, coarsening):
    original_build = legacy.build_property_data
    original_collate = legacy.collate_property_batch
    original_slice = legacy._slice_property_batch
    loaded = {}

    def build(data_root, name):
        key = (str(data_root), name)
        if key not in loaded:
            source, splits, spec = original_build(data_root, name)
            loaded[key] = (PreparedPropertyDataset(source, original_collate, cache, coarsening), splits, spec)
        return loaded[key]

    def collate(rows):
        batch = original_collate([row[:3] for row in rows])
        batch.graph = PreparedGraphBatch(batch.graph, [row[3] for row in rows])
        return batch

    def slice_batch(batch, start, end):
        sliced = original_slice(batch, start, end)
        sliced.graph = PreparedGraphBatch(sliced.graph, batch.graph.topologies[start:end])
        return sliced

    return dict(build_property_data=build, collate_property_batch=collate,
                _slice_property_batch=slice_batch)
