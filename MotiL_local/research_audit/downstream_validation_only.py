"""Author ESOL training with validation selection and no test prediction.

Cuts final test/ensemble scoring as well as per-epoch test scoring. Runtime
prediction guards reject any test-split inference. Shared patch for all groups.
"""
import ast
import csv
import importlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
from argparse import Namespace
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent/'MotiL_micromolecule'
sys.path.insert(0, str(MICRO))
import chemprop
from chemprop.parsing import parse_train_args, modify_train_args
from chemprop.torchlight import initialize_exp


def main():
    RDLogger.DisableLog('rdApp.*')
    torch.set_num_threads(2)
    seed = int(os.environ['MOTIL_RUN_SEED'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    args = parse_train_args()
    modify_train_args(args)
    if args.ensemble_size != 1 or args.num_runs != 1:
        raise ValueError('Validation-only entry requires one model per invocation')
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    runtime = importlib.import_module('chemprop.train.run_training')
    evaluate_module = importlib.import_module('chemprop.train.evaluate')
    counts = {'test_prediction_attempts': 0, 'allowed_predictions': 0}
    test_set = None
    def guard_predict(*positional, **keyword):
        nonlocal test_set
        if test_set is None:
            with (Path(args.save_dir)/'test_smiles.csv').open(newline='') as f:
                test_set = {row['smiles'] for row in csv.DictReader(f)}
        data = keyword.get('data', positional[1] if len(positional)>1 else None)
        if data is None:
            raise ValueError('Unknown prediction call signature')
        if set(data.smiles()) & test_set:
            counts['test_prediction_attempts'] += 1
            raise AssertionError('Test-split prediction prohibited in phase 4')
        counts['allowed_predictions'] += 1
        return original_predict(*positional, **keyword)
    original_predict = runtime.predict
    runtime.predict = guard_predict
    evaluate_module.predict = guard_predict
    tree = ast.parse(Path(runtime.__file__).read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_training')
    outer = next(n for n in function.body if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == 'model_idx')
    epoch_loop = next(n for n in ast.walk(outer) if isinstance(n, ast.For)
                      and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    replacements = 0
    for node in ast.walk(epoch_loop):
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == 'test_preds':
                node.value = ast.Constant(value=None); replacements += 1
            elif node.targets[0].id == 'test_scores':
                node.value = ast.parse("[float('nan')] * args.num_tasks", mode='eval').body; replacements += 1
            elif node.targets[0].id == 'avg_test_score':
                node.value = ast.Constant(value=None)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == 'info' and call.args and isinstance(call.args[0], ast.JoinedStr):
                if 'train:val:test' in ast.unparse(call.args[0]):
                    call.args[0] = ast.parse("f'Epoch: {epoch}, {args.metric} (train:val) = {avg_train_score:.6f}, {avg_val_score:.6f}'", mode='eval').body
    if replacements != 2:
        raise RuntimeError('Epoch source changed; inspect safety patch')
    reload_index = next(i for i, n in enumerate(outer.body) if isinstance(n, ast.Assign)
                        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                        and n.value.func.id == 'load_checkpoint')
    tail = ast.parse('''
validation_scores = evaluate(model=model, data=val_data, num_tasks=args.num_tasks,
    metric_func=metric_func, batch_size=args.batch_size, dataset_type=args.dataset_type,
    scaler=scaler, logger=logger)
reloaded_validation = float(np.nanmean(validation_scores))
if not np.isfinite(reloaded_validation) or abs(reloaded_validation-best_score) > 1e-5:
    raise AssertionError('Validation checkpoint reload mismatch')
writer.close()
return {'best_validation_rmse': float(best_score), 'reloaded_validation_rmse': reloaded_validation,
    'best_epoch_zero_based': int(best_epoch), 'epochs': args.epochs,
    'checkpoint_path': os.path.join(save_dir, 'model.pt'),
    'split_directory': args.save_dir, 'test_evaluated': False}
''').body
    outer.body = outer.body[:reload_index+1]+tail
    function.body = function.body[:function.body.index(outer)+1]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(runtime.__file__), 'exec'), vars(runtime))
    print('Effective PyTorch initialization seed:', seed, flush=True)
    print('Phase 4: all test predictions and scoring prohibited.', flush=True)
    result = runtime.run_training(args, logger, args.seed)
    result.update({'initialization_seed': seed, 'split_seed': args.seed, 'prediction_guard': counts})
    assert counts['test_prediction_attempts'] == 0
    assert counts['allowed_predictions'] == 2*args.epochs+1
    report = Path(os.environ['MOTIL_VALIDATION_REPORT'])
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Validation-only completed:', result['best_validation_rmse'], flush=True)


if __name__ == '__main__':
    main()
