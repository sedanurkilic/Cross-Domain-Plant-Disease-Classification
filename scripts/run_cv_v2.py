import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from fewshot_kat_v2 import run_cv

if __name__ == '__main__':
    device = torch.device('mps') if torch.backends.mps.is_available() \
             else torch.device('cpu')
    print(f"Device: {device}")
    best_epoch = run_cv(device=device, n_folds=5, num_epochs=60)
    print(f"\nBest epoch selected by CV: {best_epoch}")
    print("Next: retrain finetune_full(num_epochs=best_epoch), then run AdaBN.")
