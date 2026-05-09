import json
import random
from pathlib import Path
import os
from typing import Tuple

import torch
import torch.nn as nn
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


def train_fewshot(model: KATModelV2, train_loader: DataLoader, val_loader: DataLoader, device, save_path: str, optimizer=None):
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights_from_loader(train_loader).to(device))
    if optimizer is None:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=config.FINETUNE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        model.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / max(1, len(train_losses))

        model.eval()
        val_losses, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_losses.append(criterion(logits, labels).item())
                all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())

        avg_val_loss = sum(val_losses) / max(1, len(val_losses))
        val_acc = accuracy_score(all_targets, all_preds)
        val_bal = balanced_accuracy_score(all_targets, all_preds)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch}/{config.FINETUNE_EPOCHS} — train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f} val_bal={val_bal:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.EARLY_STOPPING:
                print("Early stopping triggered.")
                break


def evaluate_and_save(model: KATModelV2, test_loader: DataLoader, device, out_prefix: str):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
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


def finetune_full(device=None):
    """Fine-tune KATModelV2 on the entire PlantDoc finetune pool (no few-shot cap).

    Differential lr: backbone blocks.4 at 1e-5, head (agent_queries + out_proj +
    classifier) at 1e-4. blocks.4 is the hook extraction point in KATv2, so it is
    the only backbone stage whose gradients flow through the loss.
    Prototype init applied if models/kat_v2_prototype_init.pt exists.
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

    train_loader, val_loader = split_finetune_loader(full_loader, seed=config.SEED, val_frac=0.2)

    ckpt_path = Path('models/kat_v2_plantvillage_best.pth')
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found at {ckpt_path}. Run src/train_kat_v2.py first.")

    model = KATModelV2(num_classes=config.NUM_CLASSES,
                       agent_dim=config.KAT_AGENT_DIM)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    proto_path = Path('models/kat_v2_prototype_init.pt')
    if proto_path.exists():
        proto = torch.load(proto_path, map_location=device)
        model.agent_queries.data.copy_(proto['agent_queries'].to(device))
        print(f"Prototype init loaded: {proto_path}")
    else:
        print(f"WARNING: {proto_path} not found — using random agent_queries init")

    # Freeze all, then unfreeze blocks.4 (extraction point) + head
    for p in model.parameters():
        p.requires_grad = False
    model.agent_queries.requires_grad = True
    for p in model.out_proj.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    for name, param in model.named_parameters():
        if 'blocks.4' in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable} / {total}")

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'blocks.4' in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'blocks.4' not in n]
    print(f"  backbone group (lr=1e-5): {sum(p.numel() for p in backbone_params)} params")
    print(f"  head group     (lr=1e-4): {sum(p.numel() for p in head_params)} params")

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params,     'lr': 1e-4},
    ], weight_decay=1e-5)

    save_path = 'models/kat_v2_plantdoc_full_best.pth'
    train_fewshot(model, train_loader, val_loader, device, save_path, optimizer=optimizer)

    model.load_state_dict(torch.load(save_path, map_location=device))
    evaluate_and_save(model, test_loader, device, out_prefix='plantdoc_full_kat_v2')
