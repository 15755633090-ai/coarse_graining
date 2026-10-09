"""Isolated, training-compatible B0/B1/B2 motif ablations.

Import after the legacy chemprop runtime has been configured. No global source
patching, optimizer creation or changes to molecular forward are performed.
"""
import ast
from pathlib import Path
from types import MethodType
from rdkit import Chem

VARIANTS = ('B0', 'B1', 'B2a-1', 'B2a-2', 'B2b')


def choose_motifs(molecules, scopes, patterns, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    selected = {}
    if len(molecules) != len(scopes):
        raise ValueError('Molecule/scope count mismatch')
    for molecule, (offset, size) in zip(molecules, scopes):
        if molecule.GetNumAtoms() != size:
            raise ValueError('Atom count mismatch')
        for name, pattern in patterns:
            matches = molecule.GetSubstructMatches(pattern)
            if not matches or len(matches[0]) <= 1:
                continue
            if variant in ('B0', 'B1', 'B2a-1'):
                # Literal legacy batch rule, including early break at ten types.
                if len(selected) > 9:
                    break
                if name in selected:
                    continue
            elif variant == 'B2a-2' and name not in selected and len(selected) >= 10:
                # Keep all instances of admitted types, even after cap reached.
                continue
            chosen = matches[:1] if variant in ('B0', 'B1') else matches
            selected.setdefault(name, []).extend([[offset + a for a in m] for m in chosen])
    return selected


class MotifBatch:
    def __init__(self, batch, motifs):
        self.batch, self.all_fgs = batch, motifs

    def get_components(self):
        return (*self.batch.get_components()[:7], self.all_fgs)


def make_batch(smiles, args, patterns, variant):
    from chemprop.features import mol2graph
    batch = mol2graph(smiles, args)
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in molecules):
        raise ValueError('Invalid SMILES')
    return MotifBatch(batch, choose_motifs(molecules, batch.a_scope, patterns, variant))


def install_variant(encoder, variant, autograd_compatible=True):
    """Compile the snapshot forward; change only the motif loop variable.

    B0 is unchanged. Other variants rename the target and uses *inside* the
    motif message loop, leaving the molecular branch and all weights intact.
    This avoids rewriting mathematical operations while preserving a verifiable
    reference. Graph branch depth-loop behavior is unchanged intentionally.
    """
    if variant not in VARIANTS:
        raise ValueError(variant)
    import chemprop.models.cmpn as runtime
    path = Path(__file__).resolve().parent / 'upstream/cmpn.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CMPNEncoder')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    branch = next(n for n in method.body if isinstance(n, ast.If) and n.orelse)
    if variant != 'B0':
        loops = [n for node in branch.orelse for n in ast.walk(node)
                 if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'depth']
        if len(loops) != 1:
            raise ValueError('Upstream motif loop changed; review transformation')
        loop = loops[0]
        loop.target.id = 'layer_index'
        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == 'depth':
                    node.id = 'layer_index'
                return node
        loop.body = [Rename().visit(n) for n in loop.body]
    if autograd_compatible:
        class CloneBeforeWrite(ast.NodeTransformer):
            def visit_Assign(self, node):
                self.generic_visit(node)
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                    target = node.targets[0].value
                    if isinstance(target, ast.Name) and target.id in ('message_atom', 'message_bond'):
                        clone = ast.parse(f'{target.id} = {target.id}.clone()').body[0]
                        return [clone, node]
                return node
        method = CloneBeforeWrite().visit(method)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    environment = dict(vars(runtime))
    exec(compile(module, str(path), 'exec'), environment)
    encoder.forward = MethodType(environment['forward'], encoder)
    return encoder


def motif_views(encoder, batch, depth, dropout1=0.3, dropout2=0.3):
    """One vector per type, equal type contribution to original PointwiseLoss.

    Skip jointly when fewer than two types: original sample std is undefined
    for one row. Caller must log skip counts uniformly across all variants.
    No silent replacement of loss or arbitrary merging of sample axes.
    """
    if len(batch.all_fgs) < 2:
        return None
    return (encoder('pretrain', batch, 'contrast_fgs', depth, dropout1),
            encoder('pretrain', batch, 'contrast_fgs', depth - 1, dropout2))
