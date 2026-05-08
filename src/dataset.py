import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# PlantVillage için transform - domain-aware augmentation
def get_train_transform():
    return A.Compose([
        A.RandomResizedCrop(size=(config.IMAGE_SIZE, config.IMAGE_SIZE), scale=(0.6, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=5),
        ], p=0.3),
        A.RandomShadow(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.4
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class LeafDataset(Dataset):
    def __init__(self, samples, transform=None):
        # samples: [(image_path, class_idx), ...]
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


def load_plantvillage_samples(root_dir):
    """
    PlantVillage klasöründen tüm görüntü yollarını ve etiketlerini okur.
    tomato_target_spot dahil CLASS_MAPPING dışındaki klasörler atlanır.
    """
    samples = []
    valid_classes = list(config.CLASS_MAPPING.keys())

    for class_name in sorted(os.listdir(root_dir)):
        if class_name not in valid_classes:
            continue
        class_idx = config.CLASS_NAMES.index(class_name)
        class_dir = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(class_dir, fname), class_idx))
    return samples


def split_plantvillage(samples, seed=config.SEED):
    """
    PlantVillage örneklerini %80 train / %20 val olarak böler.
    Split augmentation uygulanmadan önce yapılır.
    Her sınıf kendi içinde bölünür - oranlar korunur.
    """
    set_seed(seed)
    train_samples, val_samples = [], []

    # Sınıf bazında grupla
    class_buckets = {}
    for path, label in samples:
        class_buckets.setdefault(label, []).append((path, label))

    for label, items in class_buckets.items():
        random.shuffle(items)
        cut = int(len(items) * config.TRAIN_VAL_SPLIT)
        train_samples.extend(items[:cut])
        val_samples.extend(items[cut:])

    return train_samples, val_samples


def load_plantdoc_samples(root_dir):
    """
    PlantDoc klasöründen tüm görüntü yollarını ve etiketlerini okur.
    tomato_leaf_two_spotted_spider_mites atlanır.
    PlantDoc klasör adları CLASS_MAPPING değerleri üzerinden etiketlenir.
    """
    samples = []
    # PlantDoc klasör adı -> class index eşlemesi
    plantdoc_to_idx = {
        v: config.CLASS_NAMES.index(k)
        for k, v in config.CLASS_MAPPING.items()
    }

    for folder_name in sorted(os.listdir(root_dir)):
        if folder_name not in plantdoc_to_idx:
            continue
        class_idx = plantdoc_to_idx[folder_name]
        class_dir = os.path.join(root_dir, folder_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(class_dir, fname), class_idx))
    return samples


def split_plantdoc(samples, seed=config.SEED):
    """
    PlantDoc örneklerini %20 finetune / %80 test olarak böler.
    Split augmentation uygulanmadan önce ve herhangi bir işlem yapılmadan yapılır.
    Her sınıf kendi içinde bölünür.
    """
    set_seed(seed)
    finetune_samples, test_samples = [], []

    class_buckets = {}
    for path, label in samples:
        class_buckets.setdefault(label, []).append((path, label))

    for label, items in class_buckets.items():
        random.shuffle(items)
        cut = int(len(items) * config.PLANTDOC_FINETUNE_RATIO)
        # Few-shot: finetune tarafından en fazla N örnek al
        # N değeri fewshot_finetune.py tarafından dışarıdan verilir
        finetune_samples.extend(items[:cut])
        test_samples.extend(items[cut:])

    return finetune_samples, test_samples


def get_few_shot_samples(finetune_samples, n_shot, seed=config.SEED):
    """
    Finetune havuzundan her sınıf için tam olarak n_shot örnek seçer.
    Eğer bir sınıfta n_shot kadar örnek yoksa tümünü alır ve uyarı verir.
    """
    set_seed(seed)
    class_buckets = {}
    for path, label in finetune_samples:
        class_buckets.setdefault(label, []).append((path, label))

    selected = []
    for label, items in class_buckets.items():
        random.shuffle(items)
        if len(items) < n_shot:
            print(f"Uyari: sinif {config.CLASS_NAMES[label]} icin yalnizca "
                  f"{len(items)} ornek var, {n_shot}-shot istenemez.")
        selected.extend(items[:n_shot])
    return selected


def get_plantvillage_loaders(seed=config.SEED):
    all_samples = load_plantvillage_samples(config.PLANTVILLAGE_DIR)
    train_samples, val_samples = split_plantvillage(all_samples, seed)

    train_dataset = LeafDataset(train_samples, transform=get_train_transform())
    val_dataset   = LeafDataset(val_samples,   transform=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True,  num_workers=config.NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS)
    return train_loader, val_loader


def get_plantdoc_loaders(n_shot, seed=config.SEED):
    all_samples = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, test_samples = split_plantdoc(all_samples, seed)
    few_shot_samples = get_few_shot_samples(finetune_pool, n_shot, seed)

    finetune_dataset = LeafDataset(few_shot_samples, transform=get_train_transform())
    test_dataset     = LeafDataset(test_samples,     transform=get_val_transform())

    finetune_loader = DataLoader(finetune_dataset, batch_size=min(n_shot, 8),
                                 shuffle=True,  num_workers=config.NUM_WORKERS)
    test_loader     = DataLoader(test_dataset,   batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=config.NUM_WORKERS)
    return finetune_loader, test_loader