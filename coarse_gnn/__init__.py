"""Legacy and V2 molecular graph coarsening."""
from .config import CoarseningConfig, NetworkConfig
from .cache import TopologyCache
from .model import CoarseGraphPredictor, GraphOutput
from .topology import CoarseTopology, build_topology, build_v2_topology
from .v2 import V2CoarseningConfig, V2NetworkConfig, V2CoarseGraphPredictor

__all__ = [
    "CoarseningConfig", "NetworkConfig", "CoarseGraphPredictor", "GraphOutput",
    "CoarseTopology", "build_topology", "TopologyCache",
    "V2CoarseningConfig", "V2NetworkConfig", "V2CoarseGraphPredictor",
    "build_v2_topology",
]
