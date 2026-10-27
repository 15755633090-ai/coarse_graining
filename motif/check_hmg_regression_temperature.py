"""Verify the unmodified official InfoNCE uses the SI regression temperature."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .data import save_json
from .hmg_protocol_v2 import apply_regression_settings


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('research_code_audit/HMG-MRL/model/bipartite_transformer.py'))
    a=p.parse_args();torch.set_num_threads(2)
    cfg=SimpleNamespace(device=torch.device('cpu'),temperature=.15,ct_loss_coff=999.)
    apply_regression_settings(cfg)
    assert cfg.temperature==.30 and cfg.ct_loss_coff==.15
    tree=ast.parse(a.source.read_text(encoding='utf-8-sig'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='info_nce_loss')
    env={'torch':torch,'F':F,'cfg':cfg}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(a.source),'exec'),env)
    torch.manual_seed(54);features=torch.randn(8,12,requires_grad=True)
    actual=env['info_nce_loss'](features,4)
    def manual(tau):
        normalized=F.normalize(features,dim=1)
        logits=normalized@normalized.T/tau
        logits=logits.masked_fill(torch.eye(8,dtype=torch.bool),float('-inf'))
        return F.cross_entropy(logits,(torch.arange(8)+4)%8)
    expected=manual(.30);classification=manual(.15)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    actual.backward();assert torch.isfinite(features.grad).all()
    assert abs(float(actual.detach()-classification.detach()))>1e-4
    report=dict(passed=True,tau=cfg.temperature,alpha=cfg.ct_loss_coff,
                actual_official_infonce=float(actual.detach()),manual_tau_030=float(expected.detach()),
                manual_tau_015=float(classification.detach()),
                loss_error=float((actual-expected).abs().detach()),gradient_finite=True,
                source='unchanged official info_nce_loss AST',model_training=False,test_forward=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);save_json(a.output,report);print(report)


if __name__=='__main__':main()
