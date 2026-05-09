"""
Compute per-class prototype vectors from the PlantDoc finetune pool and save
them as initial values for KATModel.agent_queries.

Usage (from project root):
    python scripts/init_kat_prototypes.py

Output:
    models/kat_prototype_init.pt  —  {'agent_queries': Tensor(8, 256)}
"""
import sys
from pathlib import Path

# Allow imports from src/
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
from model_kat import KATModel

N_SHOT = 5


def extract_spatial_features(model: KATModel, loader: DataLoader, device) -> dict[int, list]:
    """Run backbone + proj on each image, apply GAP, return {class_idx: [vec, ...]}."""
    class_feats: dict[int, list] = {}

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = model.backbone.forward_features(imgs)   # (B, 1280, 7, 7)
            proj  = model.proj(feats)                       # (B, 256, 7, 7)
            pooled = F.adaptive_avg_pool2d(proj, 1)         # (B, 256, 1, 1)
            pooled = pooled.squeeze(-1).squeeze(-1)         # (B, 256)

            for vec, label in zip(pooled.cpu(), labels):
                cls = int(label)
                class_feats.setdefault(cls, []).append(vec)

    return class_feats


def compute_prototypes(class_feats: dict[int, list]) -> torch.Tensor:
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

    # --- Load model ---
    ckpt = Path("models/kat_plantvillage_best.pth")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model = KATModel(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    print(f"Loaded checkpoint: {ckpt}")

    # --- Build deterministic dataset (val_transform, no augmentation) ---
    all_samples     = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, _= split_plantdoc(all_samples, seed=config.SEED)
    few_shot        = get_few_shot_samples(finetune_pool, n_shot=N_SHOT, seed=config.SEED)

    dataset = LeafDataset(few_shot, transform=get_val_transform())
    loader  = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=config.NUM_WORKERS)
    print(f"Dataset size: {len(dataset)} ({N_SHOT}-shot × {config.NUM_CLASSES} classes)")

    # --- Extract features and compute prototypes ---
    print("Extracting features...")
    class_feats = extract_spatial_features(model, loader, device)

    print("Computing prototypes:")
    prototypes = compute_prototypes(class_feats)   # (8, 256)
    print(f"Prototype tensor shape: {prototypes.shape}")

    # --- Save ---
    out_path = Path("models/kat_prototype_init.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent_queries": prototypes}, out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
