"""Author downstream entry, with per-epoch test evaluation disabled.

Final test evaluation remains after validation checkpoint selection. Common to
all variants. This is an isolated runtime AST patch, not a source rewrite.
"""
import ast
import importlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent/'MotiL_micromolecule'
sys.path.insert(0, str(MICRO))
import chemprop
runtime = importlib.import_module('chemprop.train.run_training')
tree = ast.parse(Path(runtime.__file__).read_text(encoding='utf-8'))
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_training')
replacements = 0
epoch_loop = next(n for n in ast.walk(function) if isinstance(n, ast.For)
                  and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
for node in ast.walk(epoch_loop):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        target = node.targets[0].id
        if target == 'test_preds' and isinstance(node.value, ast.Call) and node.value.func.id == 'predict':
            node.value = ast.Constant(value=None)
            replacements += 1
        elif target == 'test_scores' and isinstance(node.value, ast.Call) and node.value.func.id == 'evaluate_predictions':
            node.value = ast.parse("[float('nan')] * args.num_tasks", mode='eval').body
            replacements += 1
if replacements != 2:
    raise RuntimeError('Downstream source changed; review test-evaluation patch')
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
             str(runtime.__file__), 'exec'), vars(runtime))
importlib.import_module('chemprop.train.cross_validate').run_training = runtime.run_training
print('Epoch test scoring disabled; final validation-selected evaluation retained.', flush=True)
runpy.run_path(str(MICRO/'seeded_train.py'), run_name='__main__')
