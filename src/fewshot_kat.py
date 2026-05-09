import json
import random
from pathlib import Path
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from dataset import get_plantdoc_loaders
import config
from model_kat import KATModel
import numpy as np


def compute_class_weights_from_loader(loader):
    """Compute class weights from a DataLoader or Subset loader.

    Returns a torch.FloatTensor of shape (NUM_CLASSES,).
    """
    class_counts = np.zeros(config.NUM_CLASSES, dtype=np.float64)
    for _, label in loader:
        # label may be a tensor, a list, or a scalar
        if hasattr(label, '__iter__') and not isinstance(label, (str, bytes)):
            for l in label:
                class_counts[int(l)] += 1
        else:
            class_counts[int(label)] += 1

    # avoid zeros
    class_counts = np.where(class_counts == 0, 1, class_counts)
    total = class_counts.sum()
    weights = total / (config.NUM_CLASSES * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def split_finetune_loader(finetune_loader: DataLoader, seed: int = 42, val_frac: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Split finetune_loader.dataset.samples into train/val indices (deterministic).

    Returns (train_loader, val_loader) sharing the same transforms and dataset class.
    """
    dataset = finetune_loader.dataset
    samples = list(range(len(dataset)))
    random.Random(seed).shuffle(samples)
    n_val = int(len(samples) * val_frac)
    val_idx = samples[:n_val]
    train_idx = samples[n_val:]

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=finetune_loader.batch_size, shuffle=True, num_workers=finetune_loader.num_workers)
    val_loader = DataLoader(val_subset, batch_size=finetune_loader.batch_size, shuffle=False, num_workers=finetune_loader.num_workers)
    return train_loader, val_loader


def set_requires_grad_for_kat(model: KATModel):
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze agent queries and classifier
    if hasattr(model, 'agent_queries'):
        model.agent_queries.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def train_fewshot(model: KATModel, train_loader: DataLoader, val_loader: DataLoader, device, save_path: str):
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights_from_loader(train_loader).to(device))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.FINETUNE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        model.train()
        train_losses = []
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / max(1, len(train_losses))

        # Validation
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
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


def evaluate_and_save(model: KATModel, test_loader: DataLoader, device, out_prefix: str):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    bal = balanced_accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)

    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal,
        'report': report,
    }
    metrics_path = Path('results/metrics')
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(metrics_path / f"{out_prefix}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix figure
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


def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    torch.manual_seed(config.SEED)

    for n_shot in getattr(config, 'FEW_SHOT_COUNTS', [5, 10]):
        print(f"Starting few-shot fine-tune for {n_shot} shots")
        finetune_loader, test_loader = get_plantdoc_loaders(n_shot)

        train_loader, val_loader = split_finetune_loader(finetune_loader, seed=config.SEED, val_frac=0.2)

        model = KATModel(num_classes=config.NUM_CLASSES, agent_dim=config.KAT_AGENT_DIM, num_agents=config.KAT_NUM_AGENTS)
        ckpt_path = Path('models/kat_plantvillage_best.pth')
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found at {ckpt_path}. Run src/train_kat.py first.")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)

        set_requires_grad_for_kat(model)

        # debug: list trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable} / {total}")

        save_path = f"models/plantdoc_{n_shot}shot_best.pth"
        train_fewshot(model, train_loader, val_loader, device, save_path)

        # Load best before evaluation
        model.load_state_dict(torch.load(save_path, map_location=device))
        evaluate_and_save(model, test_loader, device, out_prefix=f"plantdoc_{n_shot}shot_kat")


if __name__ == '__main__':
    main()
