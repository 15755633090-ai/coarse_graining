"""Explicit computation-path controls, separate from faithful native HMG fusion."""
import ast
import copy
import inspect
import textwrap

import torch
from torch import nn
from torch.nn import functional as F

from .hmg_fusion_ablation import ARMS,variant

STUDIES=('native','noaux','common-head','common-head-noaux','single-direct','single-direct-noaux','isolated')


class PredictionMLP(nn.Sequential):
    """Same head family for both base and parameter-matched controls."""
    def __init__(self,input_dim,width=128):
        super().__init__(nn.Linear(input_dim,width),nn.ReLU(),nn.Linear(width,1))
        self.input_dim=input_dim;self.width=width


def atom_path(full):
    """Reuse exact official atom propagation/readout; discard other input paths."""
    source=textwrap.dedent(inspect.getsource(type(full).forward));tree=ast.parse(source)
    fn=tree.body[0];kept=[]
    for node in fn.body:
        code=ast.unparse(node)
        if code.startswith('frag_features = F.relu('):break
        if not isinstance(node,ast.Global):kept.append(node)
    selected=[]
    for node in fn.body:
        code=ast.unparse(node)
        if code.startswith(('mol_feature_by_atom = torch.sum(',
                            'mol_context_by_atom = self.get_mol_atom_context(',
                            'mol_feature_by_atom = self.get_mol_atom_update(')):
            selected.append(node)
    if len(selected)!=3 or not any(isinstance(n,ast.For) for n in kept):
        raise ValueError('Unsupported official atom path layout')
    fn.body=kept+selected+[ast.Return(value=ast.Name(id='mol_feature_by_atom',ctx=ast.Load()))]
    unused={'frag_src_features','frag_src_valid_mask','molecule_descriptor'}
    fn.args.args=[a for a in fn.args.args if a.arg not in unused]
    generated=ast.unparse(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])))
    env=dict(type(full).forward.__globals__);exec(compile(generated,'<isolated_atom_path>','exec'),env)
    model=copy.deepcopy(full)
    model.__class__=type('IsolatedAtomEncoder',(nn.Module,),{'forward':env['forward']})
    keep={'atom_preGNN','bond_preGNN','atom_neighbor_transform','bond_neighbor_transform',
          'atom_mlps','bond_mlps','get_mol_atom_context','get_mol_atom_update'}
    for name in list(model._modules):
        if name not in keep:delattr(model,name)
    model.generated_forward=generated
    return model


class MotifPath(nn.Module):
    """Fragment tokens only; no atom cross-attention or whole-molecule input token."""
    def __init__(self,full):
        super().__init__();self.projection=copy.deepcopy(full.frag_input_projection)
        self.transformer=copy.deepcopy(full.motif_encoder)
        self.cls_token=nn.Parameter(torch.zeros(1,1,full.atom_feature_dim))

    def forward(self,features,valid):
        fragments=F.relu(self.projection(features[:,1:]))
        token=self.cls_token.expand(features.shape[0],-1,-1)
        tokens=torch.cat([token,fragments],dim=1)
        mask=torch.cat([torch.ones_like(valid[:,:1]),valid[:,1:]],dim=1)
        return self.transformer(tokens,mask)[0][:,0]


class DescriptorPath(nn.Module):
    """Descriptor MLP only; no retrieval from atom states."""
    def __init__(self,full):super().__init__();self.layers=copy.deepcopy(full.descriptor_mlp)
    def forward(self,descriptor):
        for layer in self.layers:descriptor=layer(descriptor)
        return descriptor


class IsolatedMother(nn.Module):
    def __init__(self,full):
        super().__init__();self.A=atom_path(full);self.S=MotifPath(full);self.D=DescriptorPath(full)


class IsolatedModel(nn.Module):
    """Project-designed path isolation, never labelled an author model variant."""
    def __init__(self,mother,arm,hidden,width=128):
        super().__init__();self.arm=arm
        for key in arm:setattr(self,key,copy.deepcopy(getattr(mother,key)))
        self.postGNN_classifier=PredictionMLP(hidden*len(arm),width)

    def forward(self,**inputs):
        vectors=[]
        for key in self.arm:
            if key=='A':
                arguments={k:v for k,v in inputs.items() if k not in
                           ('frag_src_features','frag_src_valid_mask','molecule_descriptor')}
                vectors.append(self.A(**arguments))
            elif key=='S':vectors.append(self.S(inputs['frag_src_features'],inputs['frag_src_valid_mask']))
            else:vectors.append(self.D(inputs['molecule_descriptor']))
        encoded=torch.cat(vectors,dim=-1);prediction=self.postGNN_classifier(encoded)
        # None is intentional: isolated comparisons have NO alignment graph.
        return None,prediction,None,None,None,None,encoded


def native_direct(full,arm):
    if len(arm)!=1:return variant(full,arm)
    model=copy.deepcopy(full)
    source=textwrap.dedent(inspect.getsource(type(full).forward))
    start=source.index('    layer_neighbor_feature = torch.stack(')
    name={'A':'mol_feature_by_atom','S':'frag_cls','D':'mol_feature_by_des'}[arm]
    generated=source[:start]+f'''    output = self.postGNN_classifier({name})
    return atom_feature, output, ct_loss, op_mol_atom, op_mol_motif, op_mol_descrip, {name}
'''
    env=dict(type(full).forward.__globals__);exec(compile(generated,'<native_single_direct>','exec'),env)
    model.__class__=type('HMGSingleDirect'+arm,(type(full),),{'forward':env['forward']})
    model.generated_forward=generated
    return model


def _head(input_dim,width,seed):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return PredictionMLP(input_dim,width)


def make_model(full,arm,study='native',width=128,isolated_mother=None,head_seed=0):
    if arm not in ARMS or study not in STUDIES:raise ValueError((arm,study))
    if study=='isolated':
        if isolated_mother is None:raise ValueError('One common isolated mother is required per seed')
        with torch.random.fork_rng(devices=[]):
            model=IsolatedModel(isolated_mother,arm,full.atom_feature_dim,width)
        model.postGNN_classifier=_head(full.atom_feature_dim*len(arm),width,head_seed)
        return model
    model=native_direct(full,arm) if study.startswith('single-direct') else variant(full,arm)
    if study.startswith('common-head'):
        model.postGNN_classifier=_head(full.atom_feature_dim,width,head_seed)
    return model


def auxiliary_weight(study):
    return 0. if study in ('noaux','common-head-noaux','single-direct-noaux','isolated') else .15


def objective(output,target,output_mean=0.,output_std=1.,alpha=.15):
    """Always balance raw-unit MSE against alignment, regardless of head scale."""
    prediction=output[1].flatten()*output_std+output_mean
    mse=(prediction-target).square().mean()
    if alpha:
        if output[2] is None:raise ValueError('Alignment requested without an alignment graph')
        total=mse+alpha*output[2]
    else:
        # Do not write +0*aux: it would create spurious zero-gradient paths.
        total=mse
    return total,mse,prediction


def capacity_audit(model,inputs,target,alpha):
    """Loss-graph connectivity, not registered or merely nonzero parameter count."""
    was_training=model.training;model.eval();output=model(**inputs)
    total,mse,_=objective(output,target,alpha=alpha)
    pairs=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    names,parameters=zip(*pairs)
    prediction_grads=torch.autograd.grad(mse,parameters,retain_graph=True,allow_unused=True)
    all_grads=torch.autograd.grad(total,parameters,allow_unused=True)
    if any(g is not None and not torch.isfinite(g).all() for g in all_grads):raise FloatingPointError('Nonfinite capacity audit gradient')
    connected=[n for n,g in zip(names,all_grads) if g is not None]
    prediction_connected=[n for n,g in zip(names,prediction_grads) if g is not None]
    result=dict(registered_parameters=sum(p.numel() for p in model.parameters()),
        objective_connected_parameters=sum(p.numel() for p,g in zip(parameters,all_grads) if g is not None),
        prediction_connected_parameters=sum(p.numel() for p,g in zip(parameters,prediction_grads) if g is not None),
        nonzero_gradient_parameters=sum(p.numel() for p,g in zip(parameters,all_grads) if g is not None and bool((g!=0).any())),
        objective_connected_names=connected,prediction_connected_names=prediction_connected,
        sample_scope='connectivity on the recorded representative training batch; not expressivity equivalence')
    model.train(was_training)
    return result


def match_capacity(model,target_parameters,inputs,target,alpha,tolerance=.01,head_seed=0):
    """Widen a SAME-FAMILY MLP; original one-linear native head is excluded."""
    head=model.postGNN_classifier
    if not isinstance(head,PredictionMLP):raise ValueError('Capacity controls require common MLP head family')
    audit=capacity_audit(model,inputs,target,alpha)
    old=sum(p.numel() for p in head.parameters())
    encoder=audit['objective_connected_parameters']-old
    desired=target_parameters-encoder
    width=max(1,round((desired-1)/(head.input_dim+2)))
    device=next(model.parameters()).device
    model.postGNN_classifier=_head(head.input_dim,width,head_seed).to(device)
    checked=capacity_audit(model,inputs,target,alpha)
    error=abs(checked['objective_connected_parameters']-target_parameters)/target_parameters
    if error>tolerance:raise ValueError(f'Cannot match active budget: {error:.3%}')
    return dict(width=width,target=target_parameters,actual=checked['objective_connected_parameters'],
                relative_error=error,metric='objective_connected_parameters',audit=checked)
