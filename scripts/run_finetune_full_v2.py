"""Run finetune_full() for KATModelV2 on entire PlantDoc finetune pool.

Usage (from project root):
    python scripts/run_finetune_full_v2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from fewshot_kat_v2 import finetune_full

if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    finetune_full(device=device, num_epochs=32)
