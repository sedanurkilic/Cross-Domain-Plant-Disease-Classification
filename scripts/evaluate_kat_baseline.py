import sys, os
# ensure src is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

import config
from model_kat import build_kat_model
from model import get_device
from dataset import load_plantdoc_samples, split_plantdoc, LeafDataset, get_val_transform
from torch.utils.data import DataLoader


def _plot_confusion_matrix(cm, label):
    fig, ax = plt.subplots(figsize=(10, 8))

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title(f"Confusion Matrix: {label}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig_path = os.path.join(config.RESULTS_DIR, "figures", f"{label}_confusion_matrix.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")


def evaluate_kat_baseline(model_path=None):
    device = get_device()

    if model_path is None:
        model_path = os.path.join(config.MODELS_DIR, "kat_plantvillage_best.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train_kat.py first.")

    model = build_kat_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load PlantDoc test split (deterministic)
    all_samples = load_plantdoc_samples(config.PLANTDOC_DIR)
    _, test_samples = split_plantdoc(all_samples)

    test_dataset = LeafDataset(test_samples, transform=get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=config.NUM_WORKERS)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=config.CLASS_NAMES,
                                   digits=4, output_dict=True)

    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=config.CLASS_NAMES, digits=4))

    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "figures"), exist_ok=True)

    metrics_path = os.path.join(config.RESULTS_DIR, "metrics", "kat_baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": float(acc),
            "balanced_accuracy": float(balanced_acc),
            "classification_report": report
        }, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, "kat_baseline")

    return acc, balanced_acc, report


if __name__ == "__main__":
    evaluate_kat_baseline()
