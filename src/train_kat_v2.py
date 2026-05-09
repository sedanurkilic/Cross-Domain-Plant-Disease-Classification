import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import config
from dataset import get_plantvillage_loaders
from model import get_device
from model_kat_v2 import build_kat_v2
from train import compute_class_weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train(save_path=None):
    device = get_device()
    print(f"Cihaz: {device}")

    train_loader, val_loader = get_plantvillage_loaders()
    print(f"Train: {len(train_loader.dataset)} goruntu")
    print(f"Val:   {len(val_loader.dataset)} goruntu")

    model = build_kat_v2()
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Egitilecek parametre sayisi: {sum(p.numel() for p in trainable_params):,}")

    # Weighted loss for class imbalance
    class_weights = compute_class_weights(train_loader).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(trainable_params,
                     lr=config.LEARNING_RATE,
                     weight_decay=config.WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    if save_path is None:
        save_path = os.path.join(config.MODELS_DIR, "kat_v2_plantvillage_best.pth")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model kaydedildi: {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING:
                print(f"Early stopping: {config.EARLY_STOPPING} epoch "
                      f"boyunca val loss iyilesmedi.")
                break

    print(f"\nEgitim tamamlandi. En iyi val loss: {best_val_loss:.4f}")
    return save_path


if __name__ == "__main__":
    train()
