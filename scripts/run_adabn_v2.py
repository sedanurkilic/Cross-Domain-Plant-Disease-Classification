"""Apply AdaBN to a KATv2 checkpoint and evaluate on the test set.

Usage (from project root):
    python scripts/run_adabn_v2.py                         # ep60 checkpoint (default)
    python scripts/run_adabn_v2.py models/kat_v2_plantdoc_ep32_best.pth
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import torch
from fewshot_kat_v2 import evaluate_with_adabn

if __name__ == '__main__':
    device = torch.device('mps') if torch.backends.mps.is_available() \
             else torch.device('cpu')
    print(f"Device: {device}")

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate_with_adabn(device=device, checkpoint_path=ckpt, n_passes=3)
