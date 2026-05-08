import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import config
from dataset import get_plantdoc_loaders
from model import build_model, freeze_backbone, get_device, get_trainable_params


def finetune(n_shot, base_model_path=None):
    device = get_device()
    print(f"Cihaz: {device}")
    print(f"\n{n_shot}-shot fine-tuning basliyor...")

    if base_model_path is None:
        base_model_path = os.path.join(config.MODELS_DIR, "plantvillage_best.pth")

    model = build_model()
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    model = model.to(device)

    freeze_backbone(model)
    print(f"Egitilecek parametre sayisi: {get_trainable_params(model):,}")

    finetune_loader, test_loader = get_plantdoc_loaders(n_shot=n_shot)

    # Finetune pool'u %80 train / %20 val olarak böl
    all_finetune = finetune_loader.dataset.samples
    random.shuffle(all_finetune)
    cut = int(len(all_finetune) * 0.8)
    train_samples = all_finetune[:cut]
    val_samples   = all_finetune[cut:]

    from dataset import LeafDataset, get_train_transform, get_val_transform
    from torch.utils.data import DataLoader

    train_dataset = LeafDataset(train_samples, transform=get_train_transform())
    val_dataset   = LeafDataset(val_samples,   transform=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=min(n_shot, 8),
                              shuffle=True,  num_workers=config.NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=min(n_shot, 8),
                              shuffle=False, num_workers=config.NUM_WORKERS)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"Test: {len(test_loader.dataset)} goruntu")

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params,
                     lr=config.FINETUNE_LR,
                     weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    save_path = os.path.join(config.MODELS_DIR,
                             f"plantdoc_{n_shot}shot_best.pth")

    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
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
        train_loss = total_loss / total
        train_acc  = correct / total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        if val_total > 0:
            val_loss = val_loss / val_total
            val_acc  = val_correct / val_total
        else:
            val_loss = train_loss
            val_acc  = train_acc

        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING:
                print(f"Early stopping devrede.")
                break

    print(f"\n{n_shot}-shot fine-tuning tamamlandi.")
    print(f"Model kaydedildi: {save_path}")
    return save_path, test_loader


def run_all():
    base_model_path = os.path.join(config.MODELS_DIR, "plantvillage_best.pth")

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Once train.py calistirin: {base_model_path} bulunamadi."
        )

    results = {}
    for n_shot in config.FEW_SHOT_COUNTS:
        save_path, test_loader = finetune(n_shot, base_model_path)
        results[n_shot] = {
            "model_path": save_path,
            "test_loader": test_loader
        }
        print("-" * 50)

    return results


if __name__ == "__main__":
    run_all()