"""Run the author training entry with explicit RNG seeds for reproducibility."""
import os
import random
import runpy
import sys
import numpy as np
import torch
import chemprop  # Import-time legacy module seeds must run before our explicit seeds.

seed = int(os.environ['MOTIL_RUN_SEED'])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
print(f'Effective PyTorch initialization seed: {torch.initial_seed()}', flush=True)
sys.argv[0] = 'train.py'
runpy.run_path('train.py', run_name='__main__')
