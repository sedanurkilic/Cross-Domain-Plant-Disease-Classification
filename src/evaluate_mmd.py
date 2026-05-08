import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import config
from model import build_model, get_device
from dataset import (
    load_plantdoc_samples, split_plantdoc
)
from finetune_mmd import run_all


# ─────────────────────────────────────────────
# TTA Transform Listesi
# ─────────────────────────────────────────────

def get_tta_transforms():
    norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    H    = config.IMAGE_SIZE
    W    = config.IMAGE_SIZE

    return [
        # 1. Orijinal
        A.Compose([
            A.Resize(height=H, width=W),
            A.Normalize(**norm), ToTensorV2()
        ]),
        # 2. Horizontal flip
        A.Compose([
            A.Resize(height=H, width=W),
            A.HorizontalFlip(p=1.0),
            A.Normalize(**norm), ToTensorV2()
        ]),
        # 3. Vertical flip
        A.Compose([
            A.Resize(height=H, width=W),
            A.VerticalFlip(p=1.0),
            A.Normalize(**norm), ToTensorV2()
        ]),
        # 4. Hafif saat yonunde donme
        A.Compose([
            A.Resize(height=H, width=W),
            A.Rotate(limit=(10, 15), p=1.0),
            A.Normalize(**norm), ToTensorV2()
        ]),
        # 5. Hafif saat yonunun tersine donme
        A.Compose([
            A.Resize(height=H, width=W),
            A.Rotate(limit=(-15, -10), p=1.0),
            A.Normalize(**norm), ToTensorV2()
        ]),
        # 6. Parlaklik artirma
        A.Compose([
            A.Resize(height=H, width=W),
            A.RandomBrightnessContrast(
                brightness_limit=(0.1, 0.2),
                contrast_limit=(0.1, 0.2), p=1.0),
            A.Normalize(**norm), ToTensorV2()
        ]),
        # 7. Center crop
        A.Compose([
            A.Resize(height=max(H, 180), width=max(W, 180)),
            A.CenterCrop(height=180, width=180),
            A.Resize(height=H, width=W),
            A.Normalize(**norm),
            ToTensorV2()
        ]),
        # 8. Horizontal flip + parlaklik
        A.Compose([
            A.Resize(height=H, width=W),
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(0.1, 0.2),
                contrast_limit=0.1, p=1.0),
            A.Normalize(**norm), ToTensorV2()
        ]),
    ]


# ─────────────────────────────────────────────
# TTA Dataset — raw numpy goruntuleri tutar
# ─────────────────────────────────────────────

class RawImageDataset(Dataset):
    """
    Goruntuleri normalize etmeden numpy array olarak tutar.
    TTA sirasinda her augmentasyon bağımsız uygulanir.
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_np = np.array(Image.open(img_path).convert("RGB"))
        return img_np, label


def tta_collate(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels


# ─────────────────────────────────────────────
# TTA Inference
# ─────────────────────────────────────────────

def tta_predict_batch(model, raw_images, device):
    """
    raw_images: list of numpy arrays (H, W, 3)
    Her augmentasyon tum batch'e uygulanir,
    softmax olasiliklari ortalanir.
    """
    transforms   = get_tta_transforms()
    all_probs    = []

    with torch.no_grad():
        for transform in transforms:
            tensors = []
            for img_np in raw_images:
                t = transform(image=img_np)["image"]
                tensors.append(t)
            batch  = torch.stack(tensors).to(device)
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)
            all_probs.append(probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs.argmax(dim=1)


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_mmd(model_path, test_samples, label="", use_tta=False):
    device = get_device()

    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    if use_tta:
        raw_dataset = RawImageDataset(test_samples)
        loader      = DataLoader(
            raw_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            collate_fn=tta_collate
        )
        for raw_images, labels in loader:
            preds = tta_predict_batch(model, raw_images, device)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    else:
        from dataset import LeafDataset, get_val_transform
        std_dataset = LeafDataset(test_samples, transform=get_val_transform())
        loader      = DataLoader(
            std_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                preds  = model(images).argmax(dim=1).cpu().numpy()
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
        output_dict=True,
        zero_division=0
    )

    suffix = "_tta" if use_tta else ""
    print(f"\n{'='*60}")
    print(f"Results: {label}{suffix}")
    print(f"{'='*60}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print()
    print(classification_report(
        all_labels, all_preds,
        target_names=config.CLASS_NAMES,
        digits=4,
        zero_division=0
    ))

    os.makedirs(os.path.join(config.RESULTS_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "figures"), exist_ok=True)

    safe_label   = f"{label}{suffix}".replace(" ", "_").replace("-", "_")
    metrics_path = os.path.join(config.RESULTS_DIR, "metrics",
                                f"{safe_label}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "accuracy": float(acc),
            "balanced_accuracy": float(balanced_acc),
            "classification_report": report
        }, f, indent=2)

    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, safe_label)

    return acc, balanced_acc, report


def _plot_confusion_matrix(cm, label):
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_norm = np.where(
        cm.sum(axis=1, keepdims=True) > 0,
        cm.astype(float) / cm.sum(axis=1, keepdims=True), 0
    )
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix: {label}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig_path = os.path.join(config.RESULTS_DIR, "figures",
                            f"{label}_confusion_matrix.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {fig_path}")


# ─────────────────────────────────────────────
# Ana calisma akisi
# ─────────────────────────────────────────────

def run_all_evaluations():
    # Test setini bir kez ayir, tum modellerde ayni set kullanilir
    all_samples = load_plantdoc_samples(config.PLANTDOC_DIR)
    _, test_samples = split_plantdoc(all_samples)
    print(f"Test set: {len(test_samples)} images (held-out)")

    summary = {}

    # Stage 1 — fine-tuning oncesi baseline
    stage1_path = os.path.join(config.MODELS_DIR,
                               "efficientnet_mmd_stage1.pth")
    if os.path.exists(stage1_path):
        # Standard
        acc, bal, _ = evaluate_mmd(
            stage1_path, test_samples,
            label="efficientnet_mmd_stage1", use_tta=False
        )
        summary["efficientnet_mmd_stage1"] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal), 4)
        }
        # TTA
        acc, bal, _ = evaluate_mmd(
            stage1_path, test_samples,
            label="efficientnet_mmd_stage1", use_tta=True
        )
        summary["efficientnet_mmd_stage1_tta"] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal), 4)
        }

    # Stage 2 — few-shot fine-tuning + label smoothing
    results = run_all()
    for n_shot, info in results.items():
        # Standard
        acc, bal, _ = evaluate_mmd(
            model_path=info["model_path"],
            test_samples=test_samples,
            label=f"efficientnet_mmd_{n_shot}shot",
            use_tta=False
        )
        summary[f"efficientnet_mmd_{n_shot}shot"] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal), 4)
        }
        # TTA
        acc, bal, _ = evaluate_mmd(
            model_path=info["model_path"],
            test_samples=test_samples,
            label=f"efficientnet_mmd_{n_shot}shot",
            use_tta=True
        )
        summary[f"efficientnet_mmd_{n_shot}shot_tta"] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal), 4)
        }

    # Onceki sonuclari ekle
    prev_paths = {
        "efficientnet": ("summary.json", {
            "efficientnet_baseline": "baseline",
            "efficientnet_5shot":    "5_shot",
            "efficientnet_10shot":   "10_shot"
        }),
        "adakat": ("adakat_summary.json", {
            "adakat_stage0":  "adakat_stage0",
            "adakat_stage1":  "adakat_stage1",
            "adakat_5shot":   "adakat_5_shot",
            "adakat_10shot":  "adakat_10_shot"
        }),
    }

    for _, (filename, mapping) in prev_paths.items():
        path = os.path.join(config.RESULTS_DIR, "metrics", filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            prev = json.load(f)
        for new_key, old_key in mapping.items():
            if old_key in prev:
                summary[new_key] = prev[old_key]

    print("\n" + "=" * 60)
    print("Full Comparison")
    print("=" * 60)

    groups = {
        "EfficientNet (baseline)": [
            "efficientnet_baseline",
            "efficientnet_5shot",
            "efficientnet_10shot"
        ],
        "EfficientNet + MMD": [
            "efficientnet_mmd_stage1",
            "efficientnet_mmd_stage1_tta",
            "efficientnet_mmd_5shot",
            "efficientnet_mmd_5shot_tta",
            "efficientnet_mmd_10shot",
            "efficientnet_mmd_10shot_tta"
        ],
        "ADA-KAT": [
            "adakat_stage0",
            "adakat_stage1",
            "adakat_5shot",
            "adakat_10shot"
        ],
    }

    for group_name, keys in groups.items():
        print(f"\n  {group_name}")
        for key in keys:
            if key in summary:
                val = summary[key]
                print(f"    {key:40s} | "
                      f"Acc: {val.get('accuracy', 0):.4f} | "
                      f"Balanced Acc: {val.get('balanced_accuracy', 0):.4f}")

    save_path = os.path.join(config.RESULTS_DIR, "metrics",
                             "efficientnet_mmd_tta_summary.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    run_all_evaluations()