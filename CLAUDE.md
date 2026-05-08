# CLAUDE.md — plant_disease_project

## Proje Kimliği
**Başlık:** Cross-Domain Tomato Leaf Disease Classification via Kernel-Guided Attention Transformers with Interpretable Feature Localization  
**Amaç:** PlantVillage (lab) → PlantDoc (saha) domain adaptation; few-shot fine-tuning ile cross-domain domates hastalığı sınıflandırması  
**Donanım:** Apple M2 Air (MPS backend) — CUDA yok, bellek kısıtlı  

---

## Proje Yapısı

```
plant_disease_project/
├── src/                        # Ana pipeline
│   ├── config.py               # Tüm sabitler buradan okunur
│   ├── dataset.py              # LeafDataset, PlantVillage/PlantDoc loaders, few-shot sampling
│   ├── model.py                # EfficientNet-B0, freeze/unfreeze fonksiyonları
│   ├── train.py                # PlantVillage eğitimi (weighted CrossEntropy)
│   ├── fewshot_finetune.py     # PlantDoc few-shot fine-tuning (5/10-shot)
│   ├── evaluate.py             # Baseline + few-shot değerlendirme, confusion matrix
│   ├── gradcam_utils.py        # GradCAM altyapısı, overlay, figure generation
│   ├── gradcam_baseline.py     # Baseline model GradCAM görseli
│   └── gradcam_fewshot.py      # Few-shot model GradCAM görseli
└── rac/                        # Retrieval-Augmented Classification (deneysel, bağımsız)
    ├── build_memory.py
    ├── retrieval.py
    ├── fusion_model.py
    ├── train_fusion.py
    └── evaluate_rac.py
```

---

## Çalışma Sırası

```bash
# 1. PlantVillage'de EfficientNet-B0 eğit
python src/train.py
# Çıktı: models/plantvillage_best.pth

# 2. Few-shot fine-tuning (5-shot ve 10-shot)
python src/fewshot_finetune.py
# Çıktı: models/plantdoc_5shot_best.pth, models/plantdoc_10shot_best.pth

# 3. Tüm modelleri değerlendir
python src/evaluate.py
# Çıktı: results/metrics/*.json, results/figures/*_confusion_matrix.png

# 4. GradCAM görselleştirme
python src/gradcam_baseline.py
python src/gradcam_fewshot.py
# Çıktı: results/figures/gradcam_*.png
```

---

## Mevcut Sonuçlar (Kıyaslama Tablosu)

| Model                  | Accuracy | Balanced Acc | Durum      |
|------------------------|----------|--------------|------------|
| EfficientNet Baseline  | 0.2943   | 0.2517       | Tamamlandı |
| EfficientNet 5-shot    | 0.3763   | 0.3770       | Tamamlandı |
| EfficientNet 10-shot   | 0.3880   | 0.3989       | Tamamlandı |

**Başarı kriteri:** 10-shot Balanced Accuracy > 0.40

---

## Mimari Kararlar ve Gerekçeleri

### Backbone
- EfficientNet-B0 (timm kütüphanesi)
- Fine-tuning'de sadece `blocks[5]`, `blocks[6]`, `conv_head`, `classifier` açık
- Freeze stratejisi: `model.py` içindeki `freeze_backbone()` / `unfreeze_for_finetuning()`

### Domain Adaptation Yaklaşımı
- Kaynak domain: PlantVillage (kontrollü lab fotoğrafları, ~13K görüntü)
- Hedef domain: PlantDoc (saha fotoğrafları, değişken ışık/açı/arka plan)
- Strateji: Few-shot fine-tuning (5-shot / 10-shot), backbone kısmen dondurulmuş

### Loss Fonksiyonu
- PlantVillage eğitimi: weighted CrossEntropyLoss (sınıf dengesizliği var)
- Few-shot fine-tuning: düz CrossEntropyLoss (az örnekle weighted loss overfitting yaratır)

### Data Augmentation
- PlantVillage (train): aggressive augmentation — RandomResizedCrop, blur, shadow, CoarseDropout
- Validation / Test: sadece Resize + Normalize

### Bölme Stratejisi
- PlantVillage: %80 train / %20 val (sınıf bazında stratified)
- PlantDoc: %20 finetune pool / %80 test (sınıf bazında stratified, seed sabit)
- Few-shot: finetune pool'dan her sınıftan tam n_shot örnek

---

## Aktif Deney Hattı

### Sıradaki: KAT Entegrasyonu
Referans: https://github.com/Zhengyushan/kat  
**Amaç:** Prototype Attention mekanizmasını mevcut pipeline'a eklemek

Planlanan değişiklikler (öncelik sırasıyla):
1. `model_kat.py` — Multi-scale feature extraction (28x28 + 7x7) + prototype attention
2. `train_kat.py` — EfficientNet weight transfer + prototype initialization
3. `fewshot_kat.py` — Sadece prototype vektörleri fine-tune edilir (backbone dondurulur)
4. `evaluate_kat.py` — KAT sonuçlarını mevcut tabloyla karşılaştır

---

## Kurallar (Claude için)

- **Her değişiklik tek bir şeyi hedefler.** Birden fazla dosyayı aynı anda değiştirme.
- **Her çalışan değişiklik = 1 commit.** Commit mesajı formatı: `feat: <ne değişti>` / `fix: <ne düzeltildi>` / `exp: <hangi deney>`
- **config.py'a dokunmadan önce sor.** Tüm sabitler oradan okunuyor, yan etkisi yüksek.
- **MPS uyumluluğunu koru.** CUDA-only operasyonlar kullanma.
- **Test etmeden öneri sunma.** Bir değişikliği önerirken nasıl test edileceğini de belirt.
- **rac/ modülüne şimdilik dokunma.** Bağımsız deney hattı, henüz entegre değil.

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

## Notlar

- `src/` içindeki scriptler `config.py`'ı doğrudan import ediyor — `src/` dizininden çalıştırılmalı
- Seed: `config.SEED` ile sabitlenmiş, tüm split'ler tekrarlanabilir
- Model dosyaları `models/` altında, sonuçlar `results/` altında
