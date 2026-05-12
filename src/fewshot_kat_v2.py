import json
import random
from pathlib import Path
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from dataset import (
    load_plantdoc_samples,
    split_plantdoc,
    LeafDataset,
    get_train_transform,
    get_val_transform,
)
import config
from model_kat_v2 import KATModelV2
import numpy as np


def compute_class_weights_from_loader(loader):
    class_counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    for _, label in loader:
        if hasattr(label, '__iter__') and not isinstance(label, (str, bytes)):
            for l in label:
                class_counts[int(l)] += 1
        else:
            class_counts[int(label)] += 1
    class_counts = np.where(class_counts == 0, 1, class_counts)
    total = class_counts.sum()
    weights = total / (config.NUM_CLASSES * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def split_finetune_loader(finetune_loader: DataLoader, seed: int = 42, val_frac: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    dataset = finetune_loader.dataset
    samples = list(range(len(dataset)))
    random.Random(seed).shuffle(samples)
    n_val = int(len(samples) * val_frac)
    val_idx   = samples[:n_val]
    train_idx = samples[n_val:]

    train_subset = Subset(dataset, train_idx)
    val_subset   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=finetune_loader.batch_size, shuffle=True,  num_workers=finetune_loader.num_workers)
    val_loader   = DataLoader(val_subset,   batch_size=finetune_loader.batch_size, shuffle=False, num_workers=finetune_loader.num_workers)
    return train_loader, val_loader


def train_fewshot(model: KATModelV2, train_loader: DataLoader, device, save_path: str,
                  optimizer=None, test_loader: DataLoader = None, eval_every: int = 10,
                  num_epochs: int = 60):
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights_from_loader(train_loader).to(device))
    if optimizer is None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=config.FINETUNE_LR, weight_decay=1e-5)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, agent_out = model(imgs)
            ce_loss  = criterion(logits, labels)
            # diversity loss — penalise cosine similarity between agent vectors
            agent_n  = F.normalize(agent_out, dim=-1)          # (B, N, D)
            sim      = torch.bmm(agent_n, agent_n.transpose(1, 2))  # (B, N, N)
            eye_mask = torch.eye(sim.shape[1], device=sim.device).bool().unsqueeze(0)
            sim      = sim.masked_fill(eye_mask, 0.0)
            div_loss = sim.sum() / (agent_out.shape[0] * sim.shape[1] * (sim.shape[1] - 1))
            loss     = ce_loss + 0.01 * div_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / max(1, len(train_losses))
        print(f"Epoch {epoch}/{num_epochs} — train_loss={avg_train_loss:.4f}")
        torch.save(model.state_dict(), save_path)

        if test_loader is not None and epoch % eval_every == 0:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for imgs, labels in test_loader:
                    logits, _ = model(imgs.to(device))
                    preds = logits.argmax(1).cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_targets.extend(labels.numpy().tolist())
            acc = accuracy_score(all_targets, all_preds)
            bal = balanced_accuracy_score(all_targets, all_preds)
            print(f"  → Epoch {epoch} test:  accuracy={acc:.4f}  balanced_accuracy={bal:.4f}")


def train_one_fold(model: KATModelV2, train_loader: DataLoader, val_loader: DataLoader,
                   device, num_epochs: int = 60):
    """Train for one CV fold; return per-epoch val metrics without touching test set."""
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights_from_loader(train_loader).to(device)
    )

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and ('blocks.6' in n or 'conv_head' in n)]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'blocks.6' not in n and 'conv_head' not in n]
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params,     'lr': 1e-4},
    ], weight_decay=1e-5)

    val_metrics = []  # [(epoch, acc, balanced_acc), ...]

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, agent_out = model(imgs)
            ce_loss  = criterion(logits, labels)
            agent_n  = F.normalize(agent_out, dim=-1)
            sim      = torch.bmm(agent_n, agent_n.transpose(1, 2))
            eye_mask = torch.eye(sim.shape[1], device=sim.device).bool().unsqueeze(0)
            sim      = sim.masked_fill(eye_mask, 0.0)
            div_loss = sim.sum() / (agent_out.shape[0] * sim.shape[1] * (sim.shape[1] - 1))
            loss     = ce_loss + 0.01 * div_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_loss = sum(train_losses) / max(1, len(train_losses))

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                logits, _ = model(imgs.to(device))
                all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                all_targets.extend(labels.numpy().tolist())
        acc = accuracy_score(all_targets, all_preds)
        bal = balanced_accuracy_score(all_targets, all_preds)
        print(f"  ep {epoch:02d}/{num_epochs} loss={avg_loss:.4f} val_acc={acc:.4f} val_bal={bal:.4f}")
        val_metrics.append((epoch, acc, bal))

    return val_metrics


def run_cv(device=None, n_folds: int = 5, num_epochs: int = 60):
    """5-fold stratified CV on finetune pool — returns best epoch by avg val balanced_acc.

    Test set is never opened during CV.
    Results saved to results/metrics/cv_results.json.
    """
    from sklearn.model_selection import StratifiedKFold

    if device is None:
        device = torch.device('mps') if torch.backends.mps.is_available() \
                 else torch.device('cpu')

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    all_samples   = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, test_samples = split_plantdoc(all_samples, seed=config.SEED)
    print(f"Finetune pool: {len(finetune_pool)} | Test (never touched): {len(test_samples)}")

    ckpt_path = Path('models/kat_v2_plantvillage_best.pth')
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")

    labels_arr = [s[1] for s in finetune_pool]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.SEED)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(finetune_pool, labels_arr)):
        print(f"\n=== Fold {fold_idx + 1}/{n_folds} "
              f"(train={len(train_idx)}, val={len(val_idx)}) ===")

        train_samples = [finetune_pool[i] for i in train_idx]
        val_samples   = [finetune_pool[i] for i in val_idx]

        train_loader = DataLoader(
            LeafDataset(train_samples, transform=get_train_transform()),
            batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
        )
        val_loader = DataLoader(
            LeafDataset(val_samples, transform=get_val_transform()),
            batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
        )

        model = KATModelV2(num_classes=config.NUM_CLASSES, agent_dim=config.KAT_AGENT_DIM)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)

        for p in model.parameters():
            p.requires_grad = False
        model.agent_queries.requires_grad = True
        for p in model.out_proj.parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
        for name, param in model.named_parameters():
            if 'blocks.6' in name or 'conv_head' in name:
                param.requires_grad = True

        metrics = train_one_fold(model, train_loader, val_loader, device,
                                 num_epochs=num_epochs)
        fold_metrics.append(metrics)

    avg_bal = np.zeros(num_epochs)
    for fm in fold_metrics:
        for (epoch, acc, bal) in fm:
            avg_bal[epoch - 1] += bal
    avg_bal /= n_folds

    best_epoch   = int(np.argmax(avg_bal)) + 1
    best_avg_bal = float(avg_bal[best_epoch - 1])

    print(f"\n=== CV Summary ===")
    print(f"Best epoch by avg val balanced_acc: {best_epoch}")
    print(f"Avg val balanced_acc at best epoch: {best_avg_bal:.4f}")
    print("\nTop-5 epochs:")
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
    out_path = results_path / 'cv_results.json'
    with open(out_path, 'w') as f:
        json.dump(cv_out, f, indent=2)
    print(f"\nCV results saved → {out_path}")
    return best_epoch


def apply_adabn(model: KATModelV2, finetune_samples: list, device, n_passes: int = 3):
    """Update BN running stats in backbone using PlantDoc finetune images.

    Runs model.train() so BN layers update their running_mean/running_var,
    but torch.no_grad() prevents any gradient computation.
    Returns model in eval() mode ready for inference.
    """
    loader = DataLoader(
        LeafDataset(finetune_samples, transform=get_val_transform()),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
    )
    model.train()
    with torch.no_grad():
        for pass_idx in range(n_passes):
            for imgs, _ in loader:
                model(imgs.to(device))
            print(f"  AdaBN pass {pass_idx + 1}/{n_passes} done")
    model.eval()
    return model


def evaluate_with_adabn(device=None, checkpoint_path: str = None,
                        out_prefix: str = None, n_passes: int = 3):
    """Load checkpoint, apply AdaBN using finetune pool, evaluate on test set."""
    if device is None:
        device = torch.device('mps') if torch.backends.mps.is_available() \
                 else torch.device('cpu')

    all_samples   = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, test_samples = split_plantdoc(all_samples, seed=config.SEED)
    print(f"AdaBN pool (finetune): {len(finetune_pool)} | Test: {len(test_samples)}")

    if checkpoint_path is None:
        checkpoint_path = 'models/kat_v2_plantdoc_ep60_best.pth'
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if out_prefix is None:
        out_prefix = f'plantdoc_full_kat_v2_{ckpt.stem}_adabn'

    model = KATModelV2(num_classes=config.NUM_CLASSES, agent_dim=config.KAT_AGENT_DIM)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    print(f"Checkpoint loaded: {ckpt}")

    print(f"Applying AdaBN ({n_passes} passes over {len(finetune_pool)} finetune images)...")
    model = apply_adabn(model, finetune_pool, device, n_passes=n_passes)

    test_loader = DataLoader(
        LeafDataset(test_samples, transform=get_val_transform()),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS
    )
    evaluate_and_save(model, test_loader, device, out_prefix=out_prefix)
    print(f"AdaBN evaluation complete → results/metrics/{out_prefix}_metrics.json")


def evaluate_and_save(model: KATModelV2, test_loader: DataLoader, device, out_prefix: str):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())

    acc     = accuracy_score(all_targets, all_preds)
    bal     = balanced_accuracy_score(all_targets, all_preds)
    report  = classification_report(all_targets, all_preds, output_dict=True)
    cm      = confusion_matrix(all_targets, all_preds)

    metrics_path = Path('results/metrics')
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(metrics_path / f"{out_prefix}_metrics.json", 'w') as f:
        json.dump({'accuracy': acc, 'balanced_accuracy': bal, 'report': report}, f, indent=2)

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig_path = Path('results/figures')
    fig_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / f"{out_prefix}_confusion_matrix.png")
    plt.close(fig)


def finetune_full(device=None, num_epochs: int = 60):
    """Fine-tune KATModelV2 on the entire PlantDoc finetune pool (no few-shot cap).

    Differential lr: backbone blocks.6 + conv_head at 1e-5, head (agent_queries +
    out_proj + classifier) at 1e-4.
    Checkpoint saved to models/kat_v2_plantdoc_ep{num_epochs}_best.pth.
    """
    if device is None:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    torch.manual_seed(config.SEED)

    all_samples = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, test_samples = split_plantdoc(all_samples, seed=config.SEED)
    print(f"Finetune pool size: {len(finetune_pool)} | Test size: {len(test_samples)}")

    finetune_dataset = LeafDataset(finetune_pool, transform=get_train_transform())
    test_dataset     = LeafDataset(test_samples,  transform=get_val_transform())

    full_loader  = DataLoader(finetune_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=config.NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,    batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS)

    ckpt_path = Path('models/kat_v2_plantvillage_best.pth')
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found at {ckpt_path}. Run src/train_kat_v2.py first.")

    model = KATModelV2(num_classes=config.NUM_CLASSES,
                       agent_dim=config.KAT_AGENT_DIM)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    # Freeze all, then unfreeze blocks.6 + conv_head (deeper backbone) + head
    for p in model.parameters():
        p.requires_grad = False
    model.agent_queries.requires_grad = True
    for p in model.out_proj.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    for name, param in model.named_parameters():
        if 'blocks.6' in name or 'conv_head' in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total}")

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and ('blocks.6' in n or 'conv_head' in n)]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'blocks.6' not in n and 'conv_head' not in n]
    print(f"  backbone group (lr=1e-5): {sum(p.numel() for p in backbone_params)} params")
    print(f"  head group     (lr=1e-4): {sum(p.numel() for p in head_params)} params")

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params,     'lr': 1e-4},
    ], weight_decay=1e-5)

    save_path = f'models/kat_v2_plantdoc_ep{num_epochs}_best.pth'
    train_fewshot(model, full_loader, device, save_path,
                  optimizer=optimizer, test_loader=test_loader, eval_every=10,
                  num_epochs=num_epochs)

    model.load_state_dict(torch.load(save_path, map_location=device))
    evaluate_and_save(model, test_loader, device,
                      out_prefix=f'plantdoc_full_kat_v2_ep{num_epochs}')
