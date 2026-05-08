import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config
from dataset import get_plantdoc_loaders
from model import (
    build_model, get_device, get_trainable_params,
    unfreeze_for_finetuning
)


def finetune_mmd(n_shot, stage1_path=None):
    """
    Stage 2: PlantDoc few-shot fine-tuning.

    - blocks[5]: LR = 1e-5 (cok dusuk, hafif backbone adaptasyonu)
    - classifier: LR = 1e-4 (normal)
    - Diger her sey dondurulur
    """
    device = get_device()
    print(f"Device: {device}")
    print(f"\n{n_shot}-shot fine-tuning (EfficientNet+MMD) starting...")

    if stage1_path is None:
        stage1_path = os.path.join(config.MODELS_DIR,
                                   "efficientnet_mmd_stage1.pth")

    if not os.path.exists(stage1_path):
        raise FileNotFoundError(
            f"Run train_mmd.py first: {stage1_path} not found."
        )

    model = build_model()
    model.load_state_dict(torch.load(stage1_path, map_location=device))
    model = model.to(device)

    unfreeze_for_finetuning(model)
    print(f"Trainable parameters: {get_trainable_params(model):,}")

    finetune_loader, test_loader = get_plantdoc_loaders(n_shot=n_shot)
    print(f"Finetune: {len(finetune_loader.dataset)} images "
          f"({n_shot} shot x {config.NUM_CLASSES} classes)")
    print(f"Test:     {len(test_loader.dataset)} images (untouched)")

    backbone_params   = [p for n, p in model.named_parameters()
                     if p.requires_grad and "blocks.5" in n]
    classifier_params = list(model.classifier.parameters())

    optimizer = Adam([
        {"params": backbone_params,   "lr": 1e-5},
        {"params": classifier_params, "lr": config.FINETUNE_LR}
    ], weight_decay=config.WEIGHT_DECAY)

    # Yeni - tum acik parametreler ayni LR ile
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params,
                 lr=config.FINETUNE_LR,
                 weight_decay=config.WEIGHT_DECAY)
    
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=0.5, patience=3)

    criterion         = nn.CrossEntropyLoss()
    best_train_loss   = float("inf")
    epochs_no_improve = 0
    save_path         = os.path.join(
        config.MODELS_DIR,
        f"efficientnet_mmd_{n_shot}shot_best.pth"
    )

    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in finetune_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total
        scheduler.step(train_loss)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        if train_loss < best_train_loss:
            best_train_loss   = train_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING:
                print("Early stopping.")
                break

    print(f"\n{n_shot}-shot fine-tuning done. Saved: {save_path}")
    return save_path, test_loader


def run_all():
    stage1_path = os.path.join(config.MODELS_DIR,
                               "efficientnet_mmd_stage1.pth")
    if not os.path.exists(stage1_path):
        raise FileNotFoundError(
            f"Run train_mmd.py first: {stage1_path} not found."
        )

    results = {}
    for n_shot in config.FEW_SHOT_COUNTS:
        save_path, test_loader = finetune_mmd(n_shot, stage1_path)
        results[n_shot] = {
            "model_path": save_path,
            "test_loader": test_loader
        }
        print("-" * 50)

    return results


if __name__ == "__main__":
    run_all()