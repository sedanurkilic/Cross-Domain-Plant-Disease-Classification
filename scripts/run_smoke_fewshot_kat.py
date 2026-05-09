import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import config

# Temporary overrides for smoke test
config.FEW_SHOT_COUNTS = [5]
config.FINETUNE_EPOCHS = 1
config.BATCH_SIZE = 2

from fewshot_kat import main

if __name__ == '__main__':
    main()
