"""Transient functional path interventions, not learned importance scores."""
from contextlib import contextmanager
from dataclasses import dataclass,replace,asdict

@dataclass(frozen=True)
class PathGates:
    internal_branch:float=1.
    context_within:float=1.
    context_cross:float=1.
    internal_readout:float=1.
    context_readout:float=1.
    boundary_mode:str='learned'

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name!='boundary_mode' and not 0<=value<=1:
                raise ValueError(name+' must be in [0,1]')
        if self.boundary_mode not in ('learned','uniform'):
            raise ValueError('Unknown boundary mode')

DEFAULT=PathGates()
def controls(module):return getattr(module,'diagnostic_path_gates',DEFAULT)
def scaled(value,scale):return value if scale==1 else value*scale

@contextmanager
def gated_paths(model,**overrides):
    """Restore nested interventions even after errors; never change weights/mode.

    Gates are runtime diagnostics, not serialized/pruned model configuration.
    Zero internal_branch removes s from boundary gates, h updates and fusion.
    Zero internal_readout removes only its direct fusion path, retaining s->h.
    """
    encoder=getattr(model,'encoder',model)
    if not hasattr(encoder,'input_dim') or not encoder.config.dual_state or not encoder.config.motif_prior:
        raise ValueError('Path diagnostics require dual-state BRICS encoding')
    gates=replace(controls(encoder),**overrides)
    modules=[encoder,*encoder.layers];sentinel=object()
    previous=[getattr(m,'diagnostic_path_gates',sentinel) for m in modules]
    try:
        for m in modules:m.diagnostic_path_gates=gates
        yield gates
    finally:
        for m,old in zip(modules,previous):
            if old is sentinel:delattr(m,'diagnostic_path_gates')
            else:m.diagnostic_path_gates=old
