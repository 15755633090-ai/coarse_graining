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
from .hierarchy_packed import PackedHierarchyLevelPlan, PackedHierarchyPlan, pack_hierarchy
from .hierarchy_model import HierarchicalPredictor, HierarchyNetworkConfig, HierarchyOutput
from .batching import AtomCountBucketBatchSampler
from .local_boundary import (
    LocalBoundaryComponentPlan,
    local_boundary_reachability,
    split_local_boundary_component,
    split_local_boundary_topology,
    temporal_reachability,
)
from .local_boundary_packed import PackedLocalBoundaryPlan, l1_edge_attributes, pack_local_boundary
from .local_boundary_model import (
    LOCAL_BOUNDARY_VARIANTS,
    LocalBoundaryNetworkConfig,
    LocalBoundaryOutput,
    LocalBoundaryPredictor,
)

__all__ = [
    "CoarseningConfig", "NetworkConfig", "CoarseGraphPredictor", "GraphOutput",
    "CoarseTopology", "build_topology", "TopologyCache",
    "V2CoarseningConfig", "V2NetworkConfig", "V2CoarseGraphPredictor",
    "build_v2_topology",
    "ComponentHierarchy", "ContextToken", "HierarchicalTopology",
    "HierarchyConfig", "HierarchyLevel", "PackedHierarchyContextPlan",
    "HierarchyCache", "build_hierarchical_topology",
    "hierarchy_input_fingerprint", "pack_hierarchy_contexts",
    "PackedHierarchyLevelPlan", "PackedHierarchyPlan", "pack_hierarchy",
    "HierarchicalPredictor", "HierarchyNetworkConfig", "HierarchyOutput",
    "AtomCountBucketBatchSampler",
    "LocalBoundaryComponentPlan", "local_boundary_reachability",
    "split_local_boundary_component", "split_local_boundary_topology",
    "temporal_reachability",
    "PackedLocalBoundaryPlan", "l1_edge_attributes", "pack_local_boundary",
    "LOCAL_BOUNDARY_VARIANTS", "LocalBoundaryNetworkConfig",
    "LocalBoundaryOutput", "LocalBoundaryPredictor",
]
