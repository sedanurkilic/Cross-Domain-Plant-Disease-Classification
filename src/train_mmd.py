import os
import torch
from torch.utils.data import DataLoader
import config
from dataset import (
    load_plantdoc_samples,
    load_plantvillage_samples,
    split_plantdoc,
    LeafDataset,
    get_val_transform,
    get_train_transform
)
from model import build_model, get_device, freeze_all_except_classifier, get_domain_features
from model import build_model, get_device, freeze_all_except_classifier, get_domain_features, mmd_loss


def get_uda_loader(finetune_pool):
    dataset = LeafDataset(finetune_pool, transform=get_train_transform())
    return DataLoader(
        dataset,
        batch_size=config.UDA_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )


def stage1_adapt():
    device = get_device()
    print(f"Device: {device}")
    print("\n" + "=" * 60)
    print("Stage 1: Unlabeled PlantDoc Adaptation (AdaBN + MMD)")
    print("=" * 60)

    stage0_path = os.path.join(config.MODELS_DIR, "plantvillage_best.pth")
    stage1_path = os.path.join(config.MODELS_DIR,
                               "efficientnet_mmd_stage1.pth")

    if not os.path.exists(stage0_path):
        raise FileNotFoundError(
            f"Stage 0 model bulunamadi: {stage0_path}\n"
            f"Once train.py calistirin."
        )

    model = build_model()
    model.load_state_dict(torch.load(stage0_path, map_location=device))
    model = model.to(device)

    # Tum parametreleri dondur
    # BN katmanlari model.train() ile running stats guncellemeye devam eder
    freeze_all_except_classifier(model)

    # UDA loader - PlantDoc unlabeled finetune havuzu
    all_plantdoc      = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, _  = split_plantdoc(all_plantdoc)
    uda_loader        = get_uda_loader(finetune_pool)

    # Source referans loader - MMD icin PlantVillage ornekleri
    all_pv_samples    = load_plantvillage_samples(config.PLANTVILLAGE_DIR)
    source_ref_loader = DataLoader(
        LeafDataset(all_pv_samples, transform=get_val_transform()),
        batch_size=config.UDA_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    print(f"UDA target:  {len(uda_loader.dataset)} unlabeled PlantDoc images")
    print(f"Source ref:  {len(source_ref_loader.dataset)} PlantVillage images")
    print(f"Stage 1 epochs: 15")
    print(f"Backbone frozen. BN running stats updating (AdaBN).")

    source_iter   = iter(source_ref_loader)
    stage1_epochs = 15

    for epoch in range(1, stage1_epochs + 1):
        # model.train() BN running stats icin gerekli
        # gradient hesabi yok, sadece forward pass
        model.train()
        total_mmd = 0.0
        count     = 0

        for uda_images, _ in uda_loader:
            uda_images = uda_images.to(device)

            try:
                src_images, _ = next(source_iter)
            except StopIteration:
                source_iter   = iter(source_ref_loader)
                src_images, _ = next(source_iter)
            src_images = src_images.to(device)

            with torch.no_grad():
                source_feats = get_domain_features(model, src_images)
                target_feats = get_domain_features(model, uda_images)

            mmd = mmd_loss(source_feats, target_feats)
            total_mmd += mmd.item()
            count     += 1

        avg_mmd = total_mmd / count if count > 0 else 0.0
        print(f"Epoch {epoch:02d} | MMD: {avg_mmd:.6f}")

    torch.save(model.state_dict(), stage1_path)
    print(f"\nStage 1 done. Saved: {stage1_path}")
    return stage1_path


if __name__ == "__main__":
    stage1_adapt()