"""5-fold CV + full retrain for EfficientNet-B0 on PlantDoc finetune pool.

Usage (from project root):
    python scripts/run_cv_efficientnet.py              # full run: 5 folds x 60 epochs
    python scripts/run_cv_efficientnet.py --smoke      # smoke: 1 fold x 2 epochs
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

import config
from model import build_model, freeze_backbone, get_device
from dataset import (
    load_plantdoc_samples,
    split_plantdoc,
    LeafDataset,
    get_train_transform,
    get_val_transform,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_loaders(train_samples, val_samples):
    train_loader = DataLoader(
        LeafDataset(train_samples, transform=get_train_transform()),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        LeafDataset(val_samples, transform=get_val_transform()),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader


def _fresh_model(device):
    ckpt_path = Path('models/plantvillage_best.pth')
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = build_model()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    freeze_backbone(model)
    return model.to(device)


# ── per-fold training ─────────────────────────────────────────────────────────

def train_one_fold_efficientnet(model, train_loader, val_loader, device,
                                num_epochs: int = 60):
    """Train for one CV fold; return per-epoch val metrics (no early stopping)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.FINETUNE_LR, weight_decay=config.WEIGHT_DECAY,
    )

    val_metrics = []  # [(epoch, acc, balanced_acc), ...]

    for epoch in range(1, num_epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = model(imgs.to(device)).argmax(1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(labels.numpy().tolist())

        acc = accuracy_score(all_targets, all_preds)
        bal = balanced_accuracy_score(all_targets, all_preds)
        print(f"  ep {epoch:02d}/{num_epochs}  val_acc={acc:.4f}  val_bal={bal:.4f}")
        val_metrics.append((epoch, acc, bal))

    return val_metrics


# ── 5-fold CV ─────────────────────────────────────────────────────────────────

def run_cv_efficientnet(device=None, n_folds: int = 5, num_epochs: int = 60):
    """5-fold stratified CV on finetune pool — returns best epoch by avg val balanced_acc.

    Test set is never opened. Results saved to results/metrics/cv_efficientnet_results.json.
    """
    if device is None:
        device = get_device()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    all_samples   = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, test_samples = split_plantdoc(all_samples, seed=config.SEED)
    print(f"Finetune pool: {len(finetune_pool)} | Test (never touched): {len(test_samples)}")

    labels_arr = [s[1] for s in finetune_pool]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(finetune_pool, labels_arr)):
        print(f"\n=== Fold {fold_idx + 1}/{n_folds} "
              f"(train={len(train_idx)}, val={len(val_idx)}) ===")

        train_samples = [finetune_pool[i] for i in train_idx]
        val_samples   = [finetune_pool[i] for i in val_idx]
        train_loader, val_loader = _make_loaders(train_samples, val_samples)

        model   = _fresh_model(device)
        metrics = train_one_fold_efficientnet(model, train_loader, val_loader,
                                              device, num_epochs=num_epochs)
        fold_metrics.append(metrics)

    avg_bal = np.zeros(num_epochs)
    for fm in fold_metrics:
        for (epoch, acc, bal) in fm:
            avg_bal[epoch - 1] += bal
    avg_bal /= n_folds

    best_epoch   = int(np.argmax(avg_bal)) + 1
    best_avg_bal = float(avg_bal[best_epoch - 1])

    print(f"\n=== CV Summary ===")
    print(f"Best epoch: {best_epoch}  (avg val balanced_acc = {best_avg_bal:.4f})")
    print("Top-5 epochs:")
    for ep in np.argsort(avg_bal)[::-1][:5]:
        print(f"  epoch {ep + 1:02d}: {avg_bal[ep]:.4f}")

    results_path = Path('results/metrics')
    results_path.mkdir(parents=True, exist_ok=True)
    cv_out = {
        'n_folds': n_folds,
        'num_epochs': num_epochs,
        'best_epoch': best_epoch,
        'best_avg_balanced_acc': best_avg_bal,
        'avg_balanced_acc_per_epoch': avg_bal.tolist(),
        'fold_metrics': [[(e, a, b) for e, a, b in fm] for fm in fold_metrics],
    }
    out_path = results_path / 'cv_efficientnet_results.json'
    with open(out_path, 'w') as f:
        json.dump(cv_out, f, indent=2)
    print(f"CV results saved → {out_path}")
    return best_epoch


# ── full retrain + test ───────────────────────────────────────────────────────

def finetune_full_efficientnet(device=None, num_epochs: int = 60):
    """Retrain on all 146 finetune samples, evaluate on test set."""
    if device is None:
        device = get_device()

    torch.manual_seed(config.SEED)

    all_samples   = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, test_samples = split_plantdoc(all_samples, seed=config.SEED)
    print(f"Finetune: {len(finetune_pool)} | Test: {len(test_samples)}")

    full_loader = DataLoader(
        LeafDataset(finetune_pool, transform=get_train_transform()),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        LeafDataset(test_samples, transform=get_val_transform()),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS,
    )

    model     = _fresh_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.FINETUNE_LR, weight_decay=config.WEIGHT_DECAY,
    )

    save_path = Path(f'models/efficientnet_plantdoc_ep{num_epochs}_best.pth')
    save_path.parent.mkdir(exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        losses = []
        for imgs, labels in full_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / max(1, len(losses))
        print(f"Epoch {epoch:02d}/{num_epochs} — train_loss={avg_loss:.4f}")
        torch.save(model.state_dict(), save_path)

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())

    acc    = accuracy_score(all_targets, all_preds)
    bal    = balanced_accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, output_dict=True)
    cm     = confusion_matrix(all_targets, all_preds)

    results_path = Path('results/metrics')
    results_path.mkdir(parents=True, exist_ok=True)
    out_path = results_path / f'efficientnet_ep{num_epochs}_metrics.json'
    with open(out_path, 'w') as f:
        json.dump({'accuracy': acc, 'balanced_accuracy': bal, 'report': report}, f, indent=2)

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig_path = Path('results/figures')
    fig_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / f'efficientnet_ep{num_epochs}_confusion_matrix.png')
    plt.close(fig)

    print(f"\n=== Test Results (ep{num_epochs}) ===")
    print(f"accuracy:          {acc:.4f}")
    print(f"balanced_accuracy: {bal:.4f}")
    print(f"Results saved → {out_path}")
    return acc, bal


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Quick smoke test: 1 fold x 2 epochs, skip full retrain')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    if args.smoke:
        print("\n[SMOKE TEST] 2 folds x 2 epochs")
        best_epoch = run_cv_efficientnet(device=device, n_folds=2, num_epochs=2)
        print(f"\nSmoke test passed. Best epoch: {best_epoch}")
    else:
        best_epoch = run_cv_efficientnet(device=device, n_folds=5, num_epochs=60)
        print(f"\nBest epoch from CV: {best_epoch}")
        print(f"\nRetraining on full finetune pool for {best_epoch} epochs...")
        acc, bal = finetune_full_efficientnet(device=device, num_epochs=best_epoch)
        print(f"\nFinal: accuracy={acc:.4f}  balanced_accuracy={bal:.4f}")
