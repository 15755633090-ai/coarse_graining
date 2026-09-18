"""Legacy and V2 molecular graph coarsening."""
from .config import CoarseningConfig, NetworkConfig
from .cache import TopologyCache
from .model import CoarseGraphPredictor, GraphOutput
from .topology import CoarseTopology, build_topology, build_v2_topology
from .v2 import V2CoarseningConfig, V2NetworkConfig, V2CoarseGraphPredictor
from .hierarchy import (
    ComponentHierarchy,
    ContextToken,
    HierarchicalTopology,
    HierarchyConfig,
    HierarchyLevel,
    PackedHierarchyContextPlan,
    build_hierarchical_topology,
    hierarchy_input_fingerprint,
    pack_hierarchy_contexts,
)
from .hierarchy_cache import HierarchyCache

__all__ = [
    "CoarseningConfig", "NetworkConfig", "CoarseGraphPredictor", "GraphOutput",
    "CoarseTopology", "build_topology", "TopologyCache",
    "V2CoarseningConfig", "V2NetworkConfig", "V2CoarseGraphPredictor",
    "build_v2_topology",
    "ComponentHierarchy", "ContextToken", "HierarchicalTopology",
    "HierarchyConfig", "HierarchyLevel", "PackedHierarchyContextPlan",
    "HierarchyCache", "build_hierarchical_topology",
    "hierarchy_input_fingerprint", "pack_hierarchy_contexts",
]
