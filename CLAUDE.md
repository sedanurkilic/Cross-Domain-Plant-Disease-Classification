# CLAUDE.md — Cross-Domain-Plant-Disease-Classification

## Proje Kimliği
**Başlık:** Cross-Domain Tomato Leaf Disease Classification via Kernel-Guided Attention Transformers with Interpretable Feature Localization  
**Amaç:** PlantVillage (lab) → PlantDoc (saha) domain adaptation; few-shot fine-tuning baseline üzerine KAT mimarisi ile %60+ accuracy hedefi  
**Donanım:** Apple M2 Air (MPS backend) — CUDA yok, bellek kısıtlı  
**Dil:** Python 3.13  

---

## Proje Yapısı

```
plant_disease_project/
├── src/                        # Ana pipeline
│   ├── config.py               # Tüm sabitler buradan okunur — dokunmadan önce sor
│   ├── dataset.py              # LeafDataset, PlantVillage/PlantDoc loaders, few-shot sampling
│   ├── model.py                # EfficientNet-B0, freeze/unfreeze fonksiyonları
│   ├── train.py                # PlantVillage eğitimi (weighted CrossEntropy)
│   ├── fewshot_finetune.py     # PlantDoc few-shot fine-tuning (5/10-shot), train/val split var
│   ├── evaluate.py             # Baseline + few-shot değerlendirme, confusion matrix
│   ├── gradcam_utils.py        # GradCAM altyapısı, overlay, figure generation
│   ├── gradcam_baseline.py     # Baseline model GradCAM görseli
│   └── gradcam_fewshot.py      # Few-shot model GradCAM görseli
├── data/
│   └── processed/
│       ├── tomato_plantvillage/  # ~280MB, kaynak domain
│       └── tomato_plantdoc/      # ~231MB, hedef domain
├── .gitignore
└── CLAUDE.md
```

**NOT:** `rac/`, `models/`, `results/`, `data/raw/`, `data/processed/tomato_fieldplant/` gitignore'da — repoda yok.

---

## Çalışma Sırası (Mevcut Pipeline)

```bash
cd src/

# 1. PlantVillage'de EfficientNet-B0 eğit
python train.py
# Çıktı: models/plantvillage_best.pth

# 2. Few-shot fine-tuning (5-shot ve 10-shot)
python fewshot_finetune.py
# Çıktı: models/plantdoc_5shot_best.pth, models/plantdoc_10shot_best.pth

# 3. Tüm modelleri değerlendir
python evaluate.py
# Çıktı: results/metrics/*.json, results/figures/*_confusion_matrix.png

# 4. GradCAM görselleştirme
python gradcam_baseline.py
python gradcam_fewshot.py
```

---

## Mevcut Sonuçlar (Kıyaslama Tablosu)

| Model                  | Accuracy | Balanced Acc | Durum      |
|------------------------|----------|--------------|------------|
| EfficientNet Baseline  | 0.2943   | 0.2517       | Tamamlandı |
| EfficientNet 5-shot    | 0.3763   | 0.3770       | Tamamlandı |
| EfficientNet 10-shot   | 0.3880   | 0.3989       | Tamamlandı |

**Hedef:** KAT mimarisi ile 10-shot Balanced Accuracy > 0.60

---

## Mimari Kararlar

### Backbone
- EfficientNet-B0 (timm kütüphanesi)
- `freeze_backbone()`: backbone tamamen dondurulur, sadece classifier açık
- `unfreeze_for_finetuning()`: blocks[5], blocks[6], conv_head, classifier açık
- `freeze_all_except_classifier()`: MMD stage 1 için, her şey dondurulur

### Domain Adaptation Yaklaşımı
- Kaynak: PlantVillage (kontrollü lab, ~13K görüntü)
- Hedef: PlantDoc (saha fotoğrafları, değişken koşullar)
- Mevcut strateji: Few-shot fine-tuning (yetersiz, baseline)
- Hedef strateji: KAT — Prototype Attention + MLP classifier

### Loss
- PlantVillage eğitimi: weighted CrossEntropyLoss
- Few-shot fine-tuning: düz CrossEntropyLoss

---

## Aktif Geliştirme: KAT Entegrasyonu

### Hedef Mimari
```
EfficientNet-B0 backbone (dondurulmuş)
    ↓
Feature Map (7x7x1280)
    ↓
Prototype Attention (N agent, her biri farklı visual pattern'e odaklanır)
    ↓  
Agent Representations (N x 256)
    ↓
MLP Classifier → sınıf logitleri
```

### Agent Mantığı
- Her agent = farklı bir hastalık belirtisi (leke, renk değişimi, doku bozulması vb.)
- Agent'lar feature map üzerinde cross-attention yaparak ilgili bölgeyi bulur
- Tüm agent çıktıları MLP'de birleştirilir → sınıf kararı

### Sıradaki Dosyalar
| Dosya | Durum | Açıklama |
|-------|-------|----------|
| `src/model_kat.py` | Yapılacak | EfficientNet + Prototype Attention + MLP |
| `src/train_kat.py` | Yapılacak | KAT eğitimi, PlantVillage üzerinde |
| `src/fewshot_kat.py` | Yapılacak | Sadece agent_queries fine-tune |
| `src/evaluate_kat.py` | Yapılacak | KAT sonuçlarını baseline ile karşılaştır |

---

## Kurallar (Claude için)

- **Her değişiklik tek bir şeyi hedefler.** Birden fazla dosyayı aynı anda değiştirme.
- **Her çalışan değişiklik = 1 commit.** Format: `feat:` / `fix:` / `exp:`
- **config.py'a dokunmadan önce sor.** Tüm sabitler oradan okunuyor.
- **MPS uyumluluğunu koru.** CUDA-only operasyonlar kullanma.
- **Test etmeden öneri sunma.** Değişikliği önerirken nasıl test edileceğini belirt.
- **Büyük fikir atlamalar yapma.** Her adımı kullanıcıyla birlikte onayla.
- **Saçmaladığında dur.** Kullanıcı müdahale edecek.

---

## Bağımlılıklar

```
torch (MPS destekli)
torchvision
timm
albumentations
scikit-learn
matplotlib
seaborn
Pillow
numpy
```

---

## Commit Geçmişi

| Commit | Açıklama |
|--------|----------|
| `init` | Temiz proje, kod + plantdoc + plantvillage |
| `fix: freeze_backbone` | Classifier açık kalacak şekilde düzeltildi |
| `fix: compute_class_weights` | Sıfır bölme koruması eklendi |
| `fix: fewshot_finetune validation` | Train/val split, early stopping val loss'a göre |
