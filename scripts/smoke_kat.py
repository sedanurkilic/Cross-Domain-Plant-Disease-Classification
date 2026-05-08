import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import config
from model_kat import KATModel


def run_smoke(batch_size=1):
    device = torch.device("mps") if torch.backends.mps.is_available() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = KATModel()
    model.to(device)
    model.eval()

    x = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    with torch.no_grad():
        logits = model(x)
        logits_attn = model(x, return_attentions=True)

    print("logits.shape:", logits.shape)
    print("logits+attn shapes:", logits_attn[0].shape, getattr(logits_attn[1], 'shape', None))
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == '__main__':
    run_smoke(batch_size=1)
