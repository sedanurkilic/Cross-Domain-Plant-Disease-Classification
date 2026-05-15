"""Visualize KATv2 agent attention maps for one test image per class.

KATv2: 16 agents, 4 attention heads, 14x14 feature map.
attn_maps shape from model: (1, 4, 16, 14, 14) — heads averaged to (16, 14, 14).
Layout: original image full-width on top, 4x4 grid of per-agent overlays below.
Checkpoint: models/kat_v2_plantdoc_full_best.pth (ep60 finetuned).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from dataset import get_plantdoc_loaders
from model_kat_v2 import KATModelV2
import config

NUM_AGENTS = 16
GRID_ROWS  = 4
GRID_COLS  = 4

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    img = arr * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def upscale_attention(attn_map, target_h, target_w):
    t  = torch.tensor(attn_map[None, None, ...], dtype=torch.float32)
    up = F.interpolate(t, size=(target_h, target_w), mode='bilinear', align_corners=False)
    up = up.squeeze().numpy()
    return (up - up.min()) / (up.max() - up.min() + 1e-8)


def select_one_per_class(test_loader):
    dataset      = test_loader.dataset
    class_to_idx = {i: None for i in range(config.NUM_CLASSES)}
    for idx in range(len(dataset)):
        lbl = int(dataset[idx][1])
        if class_to_idx[lbl] is None:
            class_to_idx[lbl] = idx
        if all(v is not None for v in class_to_idx.values()):
            break
    return class_to_idx


def main():
    device = torch.device('mps') if torch.backends.mps.is_available() \
             else torch.device('cpu')

    ckpt_path = Path('models/kat_v2_plantdoc_full_best.pth')
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = KATModelV2(num_classes=config.NUM_CLASSES, agent_dim=config.KAT_AGENT_DIM)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    _, test_loader = get_plantdoc_loaders(5)
    class_to_idx   = select_one_per_class(test_loader)

    out_dir = Path('results/figures/kat_v2_attention')
    out_dir.mkdir(parents=True, exist_ok=True)

    short_names = [n.replace('tomato_leaf_', '').replace('tomato_', '')
                   for n in config.CLASS_NAMES]

    for cls_idx, sample_idx in sorted(class_to_idx.items()):
        if sample_idx is None:
            print(f"  No sample for class {cls_idx}, skipping")
            continue

        img, label = test_loader.dataset[sample_idx]
        img_disp   = denormalize(img)
        H, W       = img_disp.shape[:2]

        with torch.no_grad():
            logits, _, attn_maps = model(img.unsqueeze(0).to(device),
                                         return_attentions=True)

        pred      = int(logits.argmax(1).item())
        # attn_maps: (1, 4, 16, 14, 14) — average over heads
        attn      = attn_maps[0].mean(0).cpu().numpy()  # (16, 14, 14)

        # Figure: 1 row original + GRID_ROWS rows of agents
        total_rows = 1 + GRID_ROWS
        fig = plt.figure(figsize=(GRID_COLS * 4, total_rows * 4))

        ax_top = plt.subplot2grid((total_rows, GRID_COLS), (0, 0),
                                  colspan=GRID_COLS, rowspan=1)
        ax_top.imshow(img_disp)
        ax_top.axis('off')
        ax_top.set_title(
            f"Class: {short_names[cls_idx]}  |  Pred: {short_names[pred]}",
            fontsize=13
        )

        for i in range(NUM_AGENTS):
            r  = 1 + (i // GRID_COLS)
            c  = i % GRID_COLS
            ax = plt.subplot2grid((total_rows, GRID_COLS), (r, c))
            up = upscale_attention(attn[i], H, W)
            ax.imshow(img_disp)
            ax.imshow(up, cmap='jet', alpha=0.5)
            ax.axis('off')
            ax.set_title(f"Agent {i}", fontsize=9)

        out_path = out_dir / f"class_{cls_idx}_{short_names[cls_idx]}.png"
        plt.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        correct = "✓" if pred == cls_idx else "✗"
        print(f"  [{correct}] class {cls_idx} ({short_names[cls_idx]}) "
              f"pred={short_names[pred]} → {out_path.name}")


if __name__ == '__main__':
    main()
