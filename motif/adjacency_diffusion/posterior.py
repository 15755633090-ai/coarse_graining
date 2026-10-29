"""Experimental Gaussian likelihood skip; not a replacement for core defaults."""
import torch
from .model import AdjacencyModel


def likelihood_log_odds(noisy, alpha, pairs):
    """log p(y|A=1)/p(y|A=0), unit Gaussian forward variance 1-alpha."""
    if torch.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError('Positive-noise alpha must be strictly between zero and one')
    return ((2*alpha.sqrt()*noisy-alpha)/(2*(1-alpha))) * pairs


class PosteriorModel(AdjacencyModel):
    def __init__(self,config,evidence=True):
        super().__init__(config)
        self.evidence=evidence

    def reconstruction_loss(self,*args,**kwargs):
        raise RuntimeError('This head returns adjacency logits; use the experimental sigmoid reconstruction objective, not epsilon inversion')

    def predict_noise(self,condition,noisy,t,return_states=False):
        # This experimental head returns clean-adjacency LOGITS, never epsilon.
        observed=noisy if self.evidence else torch.zeros_like(noisy)
        logits,states=super().predict_noise(condition,observed,t,True)
        if self.evidence:
            alpha=self.schedule.alpha_bar.to(noisy.device)[t].to(noisy.dtype)[:,None,None]
            logits=logits+likelihood_log_odds(noisy,alpha,condition.pairs)
        logits=logits*condition.pairs
        return (logits,states) if return_states else logits
