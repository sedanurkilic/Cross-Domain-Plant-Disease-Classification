"""
Compute per-class prototype vectors from the PlantDoc finetune pool using
KATModelV2 and save them as initial values for KATModelV2.agent_queries.

Usage (from project root):
    python scripts/init_kat_v2_prototypes.py

Output:
    models/kat_v2_prototype_init.pt  —  {'agent_queries': Tensor(8, 256)}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from dataset import (
    load_plantdoc_samples,
    split_plantdoc,
    get_few_shot_samples,
    LeafDataset,
    get_val_transform,
)
from model_kat_v2 import KATModelV2

N_SHOT = 5


def extract_spatial_features(model: KATModelV2, loader: DataLoader, device) -> dict:
    """Run backbone (via hook) + proj on each image, apply GAP, return {class_idx: [vec, ...]}."""
    class_feats: dict = {}

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)

            model._feat_cache.clear()
            model.backbone.forward_features(imgs)       # triggers blocks[4] hook
            feats  = model._feat_cache['x']             # (B, 112, 14, 14)
            proj   = model.proj(feats)                  # (B, 256, 14, 14)
            pooled = F.adaptive_avg_pool2d(proj, 1)     # (B, 256, 1, 1)
            pooled = pooled.squeeze(-1).squeeze(-1)     # (B, 256)

            for vec, label in zip(pooled.cpu(), labels):
                cls = int(label)
                class_feats.setdefault(cls, []).append(vec)

    return class_feats


def compute_prototypes(class_feats: dict) -> torch.Tensor:
    """Average per-class feature vectors → (NUM_CLASSES, agent_dim) tensor."""
    prototypes = []
    for cls_idx in range(config.NUM_CLASSES):
        vecs = class_feats.get(cls_idx, [])
        if not vecs:
            print(f"  WARNING: no samples for class {cls_idx} ({config.CLASS_NAMES[cls_idx]}), using zeros")
            prototypes.append(torch.zeros(config.KAT_AGENT_DIM))
        else:
            prototypes.append(torch.stack(vecs).mean(dim=0))
            print(f"  Class {cls_idx} ({config.CLASS_NAMES[cls_idx]}): {len(vecs)} samples")
    return torch.stack(prototypes)  # (8, 256)


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    ckpt = Path("models/kat_v2_plantvillage_best.pth")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = KATModelV2(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    print(f"Loaded checkpoint: {ckpt}")

    all_samples    = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, _ = split_plantdoc(all_samples, seed=config.SEED)
    few_shot       = get_few_shot_samples(finetune_pool, n_shot=N_SHOT, seed=config.SEED)

    dataset = LeafDataset(few_shot, transform=get_val_transform())
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    print(f"Dataset size: {len(dataset)} ({N_SHOT}-shot × {config.NUM_CLASSES} classes)")

    print("Extracting features...")
    class_feats = extract_spatial_features(model, loader, device)

    print("Computing prototypes:")
    prototypes = compute_prototypes(class_feats)
    print(f"Prototype tensor shape: {prototypes.shape}")

    out_path = Path("models/kat_v2_prototype_init.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent_queries": prototypes}, out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
