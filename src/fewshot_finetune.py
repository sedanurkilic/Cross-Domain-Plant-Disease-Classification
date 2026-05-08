import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import config
from dataset import get_plantdoc_loaders
from model import build_model, freeze_backbone, get_device, get_trainable_params


def finetune(n_shot, base_model_path=None):
    device = get_device()
    print(f"Cihaz: {device}")
    print(f"\n{n_shot}-shot fine-tuning basliyor...")

    # Egitilmis PlantVillage modelini yukle
    if base_model_path is None:
        base_model_path = os.path.join(config.MODELS_DIR, "plantvillage_best.pth")

    model = build_model()
    model.load_state_dict(torch.load(base_model_path, map_location=device))
    model = model.to(device)

    # Backbone'u dondur, sadece classifier head egitilecek
    freeze_backbone(model)
    print(f"Egitilecek parametre sayisi: {get_trainable_params(model):,}")

    finetune_loader, test_loader = get_plantdoc_loaders(n_shot=n_shot)
    print(f"Finetune: {len(finetune_loader.dataset)} goruntu "
          f"({n_shot} shot x {config.NUM_CLASSES} sinif)")
    print(f"Test:     {len(test_loader.dataset)} goruntu (dokunulmamis)")

    # Few-shot'ta sinif basina ornek sayisi cok az oldugu icin
    # weighted loss burada kullanmiyoruz, overfitting riskini arttirir
    criterion = nn.CrossEntropyLoss()

    # Sadece egitilecek parametreleri optimizer'a ver
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

        for images, labels in finetune_loader:
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

        # --- Validation: finetune havuzunun kendisi ---
        # Not: burada ayri bir val set kullanmiyoruz cunku n_shot zaten
        # cok kucuk. Train loss'u takip ederek overfitting'i izliyoruz.
        scheduler.step(train_loss)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        if train_loss < best_val_loss:
            best_val_loss = train_loss
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