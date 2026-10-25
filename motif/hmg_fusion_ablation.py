"""End-to-end HMG-MRL fusion-view ablations; original auxiliary objective kept."""
import copy
import inspect
import textwrap

ARMS = ('A', 'S', 'D', 'AS', 'AD', 'SD', 'ASD')


def variant(full, arm, forward_path=None):
    if arm not in ARMS:
        raise ValueError(arm)
    model = copy.deepcopy(full)
    source = textwrap.dedent(inspect.getsource(type(full).forward))
    original = 'torch.stack([frag_cls, mol_feature_by_atom, mol_feature_by_des], dim=1)'
    if source.count(original) != 1:
        raise ValueError('Unsupported HMG fusion layout')
    names = [('S','frag_cls'),('A','mol_feature_by_atom'),('D','mol_feature_by_des')]
    chosen = [name for scale,name in names if scale in arm]
    generated = source.replace(original, f"torch.stack([{', '.join(chosen)}], dim=1)")
    if forward_path is not None:
        forward_path.write_text(generated+'\n', encoding='utf-8')
    if arm != 'ASD':
        env = dict(type(full).forward.__globals__)
        exec(compile(generated, '<hmg_fusion_view_ablation>', 'exec'), env)
        derived = type(f'HMGFusion{arm}', (type(full),), {'forward': env['forward']})
        model.__class__ = derived
    # All parameters, buffers, and training objectives are retained unchanged.
    assert set(model.state_dict()) == set(full.state_dict())
    assert all((v == full.state_dict()[k]).all() for k,v in model.state_dict().items())
    return model
