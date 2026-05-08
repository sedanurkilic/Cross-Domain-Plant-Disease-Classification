import os
import json
import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import config
from model import build_model, get_device
from dataset import get_plantdoc_loaders, load_plantdoc_samples, split_plantdoc


def evaluate(model_path, test_loader, label=""):
    device = get_device()

    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc          = (all_preds == all_labels).mean()
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    report       = classification_report(
        all_labels, all_preds,
        target_names=config.CLASS_NAMES,
        digits=4,
        output_dict=True
    )

    print(f"\n{'='*60}")
    if label:
        print(f"Sonuclar: {label}")
    print(f"{'='*60}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print()
    print(classification_report(
        all_labels, all_preds,
        target_names=config.CLASS_NAMES,
        digits=4
    ))

    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "figures"), exist_ok=True)

    safe_label = label.replace(" ", "_").replace("-", "_") if label else "eval"

    metrics_path = os.path.join(config.RESULTS_DIR, "metrics",
                                f"{safe_label}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": acc,
            "balanced_accuracy": balanced_acc,
            "classification_report": report
        }, f, indent=2)

    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, safe_label)

    return acc, balanced_acc, report


def evaluate_baseline():
    """
    Fine-tuning yapilmadan PlantVillage modelini direkt PlantDoc
    test seti uzerinde degerlendirir. Bu bize domain gap'in buyuklugunu gosterir.
    """
    base_model_path = os.path.join(config.MODELS_DIR, "plantvillage_best.pth")

    # PlantDoc'u ayni seed ile boluyor, test seti her zaman ayni kaliyor
    all_samples = load_plantdoc_samples(config.PLANTDOC_DIR)
    _, test_samples = split_plantdoc(all_samples)

    from dataset import LeafDataset, get_val_transform
    from torch.utils.data import DataLoader

    test_dataset = LeafDataset(test_samples, transform=get_val_transform())
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS)

    print(f"Baseline test: {len(test_dataset)} goruntu (fine-tuning yok)")
    return evaluate(base_model_path, test_loader, label="baseline")


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

    fig_path = os.path.join(config.RESULTS_DIR, "figures",
                            f"{label}_confusion_matrix.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()


def run_all_evaluations():
    from fewshot_finetune import run_all

    base_model_path = os.path.join(config.MODELS_DIR, "plantvillage_best.pth")
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Once train.py calistirin: {base_model_path} bulunamadi."
        )

    summary = {}

    # 1. Baseline
    acc, balanced_acc, _ = evaluate_baseline()
    summary["baseline"] = {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(balanced_acc), 4)
    }

    # 2. Few-shot modelleri egit ve degerlendir
    results = run_all()
    for n_shot, info in results.items():
        acc, balanced_acc, _ = evaluate(
            model_path=info["model_path"],
            test_loader=info["test_loader"],
            label=f"plantdoc_{n_shot}shot"
        )
        summary[f"{n_shot}_shot"] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(balanced_acc), 4)
        }

    print("\n" + "="*60)
    print("Ozet Karsilastirma")
    print("="*60)
    for key, val in summary.items():
        print(f"{key:12s} | Acc: {val['accuracy']:.4f} | "
              f"Balanced Acc: {val['balanced_accuracy']:.4f}")

    summary_path = os.path.join(config.RESULTS_DIR, "metrics", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOzet kaydedildi: {summary_path}")


if __name__ == "__main__":
    run_all_evaluations()