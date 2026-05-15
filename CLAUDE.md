# CLAUDE.md — Cross-Domain Plant Disease Classification

**Son güncelleme:** 15 Mayıs 2026

---

## Proje Hakkında

Bu proje, domates yaprak hastalıklarını kontrollü laboratuvar ortamından (PlantVillage) saha fotoğraflarına (PlantDoc) taşıma problemini ele alıyor. Temel zorluk şu: lab ortamında %99'un üzerinde doğrulukla çalışan bir model, saha koşullarında dramatik biçimde başarısız oluyor. Bunu domain gap problemi olarak adlandırıyoruz.

KATv1 deneyleri tamamlandı; EfficientNet few-shot'ın gerisinde kaldı. KATv2 (16 agent, 4-head attention, 14×14 feature map) ile val split kaldırılıp sabit epoch eğitimine geçildi. Diversity loss (lambda=0.01) ile epoch 60'ta **acc 0.4565 / balanced 0.4510** elde edildi — ancak bu sonuç test setine bakılarak seçildiği için leaky.

5-fold CV uygulandı: metodolojik olarak temiz best epoch **32**, CV avg val balanced_acc **0.3729**. Epoch 32 ile yeniden eğitilen modelin gerçek test performansı: acc 0.3595 / balanced 0.3545. AdaBN, blocks.6 + conv_head açık olduğunda işe yaramadı — BN stats eğitim sırasında zaten adapte oluyor. Sıradaki: bacterial_spot ve mosaic_virus sınıflarına odaklı strateji.

Focal Loss (gamma=2) eklendi: CV ortalaması 0.3729 → 0.4025 iyileşti ancak test'e yansımadı (ep32: acc 0.3462, balanced 0.3575). KATv2 attention map analizi yapıldı — iki hata türü tespit edildi: görsel benzerlik (bacterial_spot↔septoria) ve hastalık evresi farklılığı (early_blight ileri evrede yellow_virus'a benziyor). FieldPlant veri seti ek test seti olarak değerlendiriliyor.

EfficientNet 5-fold CV tamamlandı (15 Mayıs): best epoch **20**, CV avg val balanced_acc **0.3767**, test acc 0.3729 / balanced **0.3553**. **Temel bulgu: KATv2 ve EfficientNet CV-temiz sonuçları neredeyse özdeş (0.3545 vs 0.3553). Mimari farklılık performansı açıklamıyor — asıl darboğaz 146 görüntülük finetune pool.**

---

## Proje Yapısı

```
plant_disease_project/
├── src/
│   ├── config.py               # Tüm sabitler burada — değiştirmeden önce mutlaka sor
│   ├── dataset.py              # Veri yükleme, augmentation, few-shot sampling
│   ├── model.py                # EfficientNet-B0, freeze/unfreeze yardımcıları
│   ├── model_kat.py            # KATv1 modeli (8 agent, single-head, 7x7)
│   ├── train_kat.py            # KAT PlantVillage pre-training pipeline
│   ├── train.py                # PlantVillage üzerinde baseline eğitimi
│   ├── fewshot_finetune.py     # EfficientNet PlantDoc few-shot fine-tuning
│   ├── fewshot_kat.py          # KAT few-shot + full finetune pipeline
│   ├── evaluate.py             # Tüm modellerin değerlendirilmesi
│   ├── gradcam_utils.py        # GradCAM altyapısı ve görselleştirme
│   ├── gradcam_baseline.py     # Baseline model için GradCAM
│   └── gradcam_fewshot.py      # Few-shot modeller için GradCAM
├── scripts/
│   ├── evaluate_kat_baseline.py   # KAT baseline PlantDoc değerlendirmesi
│   ├── init_kat_prototypes.py     # KATv1 PlantDoc class prototype vektörleri
│   ├── init_kat_v2_prototypes.py  # KATv2 PlantDoc class prototype vektörleri
│   ├── run_finetune_full.py       # KATv1 finetune_full() çağırır
│   ├── run_finetune_full_v2.py    # KATv2 finetune_full(num_epochs=32) çağırır
│   ├── run_cv_v2.py               # KATv2 5-fold CV — best epoch seçimi
│   ├── run_adabn_v2.py            # AdaBN değerlendirmesi (checkpoint argümanı alır)
│   ├── run_cv_efficientnet.py     # EfficientNet 5-fold CV + full retrain
│   └── visualize_kat_attention.py # Agent attention haritası görselleştirme
├── data/
│   └── processed/
│       ├── tomato_plantvillage/
│       └── tomato_plantdoc/
├── .gitignore
└── CLAUDE.md
```

`models/`, `results/`, `data/raw/`, `data/processed/tomato_fieldplant/` gitignore kapsamında — repoda bulunmuyor.

---

## Mevcut Sonuçlar

| Model                        | Accuracy | Balanced Acc | Durum      |
|------------------------------|----------|--------------|------------|
| EfficientNet Baseline        | 0.2943   | 0.2517       | Tamamlandı |
| EfficientNet 5-shot          | 0.3763   | 0.3770       | Tamamlandı |
| EfficientNet 10-shot         | 0.3880   | 0.3989       | Tamamlandı |
| KATv1 Baseline               | 0.2341   | 0.2173       | Tamamlandı |
| KATv1 5-shot                 | 0.1656   | 0.1590       | Tamamlandı |
| KATv1 10-shot                | 0.1873   | 0.2080       | Tamamlandı |
| KATv1 Full finetune (diff lr)| 0.2007   | 0.2031       | Tamamlandı |
| KATv2 Baseline               | 0.2492   | 0.2359       | Tamamlandı |
| KATv2 Full — epoch 40        | 0.4130   | 0.4025       | Tamamlandı |
| KATv2 Full + div loss — ep10 | 0.3361   | 0.3354       | Tamamlandı |
| KATv2 Full + div loss — ep20 | 0.3612   | 0.3588       | Tamamlandı |
| KATv2 Full + div loss — ep30 | 0.3946   | 0.3898       | Tamamlandı |
| KATv2 Full + div loss — ep40 | 0.4130   | 0.4074       | Tamamlandı |
| KATv2 Full + div loss — ep50 | 0.4247   | 0.4302       | Tamamlandı |
| KATv2 Full + div loss — ep60 (leaky) | 0.4565 | 0.4510 | Tamamlandı — leaky |
| **KATv2 ep32 — CV-temiz (blocks.6+conv_head)** | **0.3595** | **0.3545** | Tamamlandı |
| KATv2 ep32 + AdaBN           | 0.3512   | 0.3399       | Tamamlandı — AdaBN zararlı |
| KATv2 ep60 + AdaBN           | 0.4197   | 0.4158       | Tamamlandı — AdaBN zararlı |
| KATv2 + FocalLoss ep32       | 0.3462   | 0.3575       | Tamamlandı — CV iyileşti, test'e yansımadı |
| **EfficientNet CV-temiz ep20** | **0.3729** | **0.3553** | Tamamlandı — CV best epoch=20 |

Hedef: balanced accuracy > 0.60

**Metodoloji notu:** ep60 sonucu (0.4510) test setine bakılarak epoch seçildiği için leaky. CV ile elde edilen ep32 sonucu (0.3545) metodolojik olarak temiz; bu gerçek genelleme tahminidir.

Val split kaldırıldı — tüm 146 finetune örneği direkt train'e veriliyor. `run_finetune_full_v2.py` şu an `num_epochs=32` ile çalışıyor.

### 5-Fold CV Sonuçları (13 Mayıs 2026)

146 finetune örneği, 5-fold stratified CV. Her fold ~117 train / ~29 val.

| Fold | val_acc | val_balanced_acc |
|------|---------|-----------------|
| 1    | 0.3000  | 0.2812          |
| 2    | 0.3448  | 0.3708          |
| 3    | 0.4828  | 0.5104          |
| 4    | 0.3103  | 0.2708          |
| 5    | 0.2759  | 0.2500          |

**Best epoch: 32** (avg val balanced_acc = 0.3729)

Fold'lar arası varyans yüksek (0.25–0.51). Fold başına yalnızca ~29 val örneği düşüyor — bu ölçüm gürültüsü yaratıyor. CV fold varyansı, 146 örneklik finetune pool'unun epoch seçimi için çok küçük olduğunu gösteriyor.

### 5-Fold CV — Focal Loss (15 Mayıs 2026)

| Fold | val_acc | val_balanced_acc |
|------|---------|-----------------|
| 1    | 0.3333  | 0.3271          |
| 2    | 0.3448  | 0.3708          |
| 3    | 0.3448  | 0.3396          |
| 4    | 0.2414  | 0.1979          |
| 5    | 0.3103  | 0.3333          |

**Best epoch: 32** (avg val balanced_acc = 0.4025)

CV ortalaması +0.030 iyileşti (0.3729 → 0.4025). Best epoch yine 32 — tutarlı. Ancak test sonucu minimal değişti (balanced 0.3545 → 0.3575). CV iyileşmesi test'e yansımadı — 146 örneklik finetune pool hâlâ temel darboğaz.

### 5-Fold CV — EfficientNet Karşılaştırması (15 Mayıs 2026)

`run_cv_efficientnet.py`: `plantvillage_best.pth` yükle → `freeze_backbone()` → 146 finetune pool'da 5-fold CV × 60 epoch. CrossEntropyLoss, sabit lr.

**Best epoch: 20** (avg val balanced_acc = 0.3767)
**Test: acc=0.3729, balanced_accuracy=0.3553**

### Mimari Karşılaştırma — CV-Temiz Sonuçlar

| Model | CV Best Epoch | CV Avg Bal Acc | Test Acc | Test Balanced Acc |
|---|---|---|---|---|
| EfficientNet (CrossEntropy) | 20 | 0.3767 | 0.3729 | 0.3553 |
| KATv2 (CrossEntropy) | 32 | 0.3729 | 0.3595 | 0.3545 |
| KATv2 (FocalLoss γ=2) | 32 | 0.4025 | 0.3462 | 0.3575 |

**Temel bulgu:** EfficientNet ve KATv2 test performansı CV-temiz koşullarda neredeyse özdeş (~0.354–0.355 balanced). Mimari fark istatistiksel olarak anlamlı değil. KATv2'nin prototype attention mekanizması domain adaptation avantajı sağlamıyor — en azından bu veri miktarında.

**Asıl darboğaz:** 146 görüntülük finetune pool. Loss fonksiyonu, mimari veya regularizasyon değişikliklerinin hiçbiri bu sınırı aşamadı. Bir sonraki anlamlı iyileşme için daha fazla saha verisi gerekiyor.

### AdaBN Analizi

| Checkpoint | Baseline | + AdaBN | Fark |
|---|---|---|---|
| ep32 | 0.3545 | 0.3399 | −0.015 |
| ep60 | 0.4510 | 0.4158 | −0.035 |

**Sonuç:** AdaBN her ikisinde de performansı düşürdü. **Sebebi:** blocks.6 + conv_head trainable olduğu için BN istatistikleri eğitim sırasında PlantDoc'a zaten adapte oldu. Inference'ta tekrar güncellemek mevcut adaptasyonu bozuyor. AdaBN yalnızca tamamen dondurulmuş backbone'larda etkili — bu konfigürasyonda geçerli değil.

---

## KATv1 Mimarisi (model_kat.py)

```
EfficientNet-B0 (dondurulmuş backbone)
    → forward_features → (B, 1280, 7, 7)
    → 1x1 Conv projection → (B, 256, 7, 7)
    → spatial flatten → (B, 49, 256)
    → Cross-attention: 8 agent query × 49 spatial token (single-head)
    → agent outputs → (B, 8, 256)
    → LayerNorm → Flatten → (B, 2048)
    → MLP: 2048 → 512 → 256 → NUM_CLASSES
```

---

## KATv2 Mimarisi (model_kat_v2.py)

```
EfficientNet-B0 (backbone — blocks.6 + conv_head açık, geri kalan donuk)
    → forward hook on blocks[4] → (B, 112, 14, 14)
    → 1x1 Conv projection → (B, 256, 14, 14)
    → spatial flatten → (B, 196, 256)
    → 4-head cross-attention: 16 agent query × 196 spatial token
    → agent outputs → (B, 16, 256)
    → out_proj + LayerNorm → Flatten → (B, 4096)
    → MLP: 4096 → 512 → 256 → NUM_CLASSES
```

finetune_full() eğitim stratejisi:
- Tüm 146 finetune örneği train'e verilir (val split yok)
- `num_epochs` parametresi — şu an 32 (CV'den seçildi)
- Differential lr: backbone blocks.6 + conv_head → 1e-5, head → 1e-4
- Checkpoint: `models/kat_v2_plantdoc_ep{num_epochs}_best.pth`

---

## Deney Bulguları

**KATv1:**
- Val split çok küçük (≤29 örnek) — early stopping gürültülü, güvenilmez.
- Prototype init anlamlı iyileşme sağlamadı — backbone PlantVillage'e kilitli olduğundan feature uzayı uyumsuz.
- Kök sorun: 7×7 feature map + single-head attention saha görüntüleri için yetersiz.

**KATv2:**
- Val split kaldırınca ve sabit epoch eğitimine geçince performans tutarlı biçimde arttı.
- Prototype tiling (8→16) eğitimi bozdu — epoch 1'de loss patladı, kullanılmıyor.
- Diversity loss (lambda=0.01) epoch 50'de balanced acc'ı 0.4290'a taşıdı (önceki best: 0.4025 @ epoch 40).
- Epoch 60'ta acc 0.4565 / balanced 0.4510 — leaky (test setine bakılarak seçildi).
- **5-fold CV (13 Mayıs):** best epoch = 32, temiz test sonucu acc 0.3595 / balanced 0.3545. Leakage'ın etkisi: +0.097 balanced acc şişme.
- **AdaBN (13 Mayıs):** blocks.6 + conv_head açık olduğunda zararlı. BN stats eğitimde zaten adapte oluyor, inference'ta tekrar güncelleme bozuyor.
- freeze stratejisi: blocks.4 → blocks.6 + conv_head olarak değiştirildi (13 Mayıs).
- **Focal Loss gamma=2 (15 Mayıs):** CV ortalaması +0.030 iyileşti ama test'e yansımadı. Temel sorun loss fonksiyonu değil, finetune pool büyüklüğü (146 örnek).

---

## Sınıf Bazlı Analiz (ep60 checkpoint, 598 test örneği)

| Sınıf | n | Precision | Recall | F1 | Durum |
|---|---|---|---|---|---|
| bacterial_spot | 88 | 0.308 | 0.136 | **0.189** | En kötü |
| early_blight | 71 | 0.389 | 0.521 | 0.446 | Orta |
| healthy | 51 | 0.368 | 0.490 | 0.420 | Orta |
| late_blight | 89 | 0.586 | 0.730 | **0.650** | En iyi |
| mold | 73 | 0.500 | 0.219 | 0.305 | Kötü — recall düşük |
| mosaic_virus | 44 | 0.255 | 0.318 | 0.283 | Kötü — az veri |
| yellow_virus | 61 | 0.586 | 0.672 | 0.626 | İyi |
| septoria_leaf_spot | 121 | 0.492 | 0.521 | 0.506 | Orta |

**Bulgular:**
- **late_blight (F1=0.650) ve yellow_virus (F1=0.626):** Görsel olarak ayırt edici semptomlar saha fotoğraflarında da korunuyor.
- **bacterial_spot (F1=0.189):** n=88 ile büyük sınıf ama recall=0.14 — örneklerin %86'sı yanlış etiketleniyor. Septoria ile görsel karışıklık muhtemel (her ikisi de küçük koyu leke).
- **mold (F1=0.305):** Precision=0.50 yüksek ama recall=0.22 çok düşük — model mold'u tanıdığında haklı ama çoğunu kaçırıyor. Dominant sınıflara (septoria, late_blight) akıyor olabilir.
- **mosaic_virus (F1=0.283):** En az örnekli sınıf (n=44) — finetune pool'da yetersiz görüldü.

---

## Sıradaki Adımlar

### a) Temel Darboğaz: Veri Miktarı

Mimari ve loss fonksiyonu deneyleri sonuca göre: **mevcut 146 görüntülük finetune pool ile ~0.35 balanced accuracy tavanı var.** Bunu aşmak için daha fazla saha verisi şart.

**Seçenekler (öncelik sırasıyla):**

**1. Ek saha verisi toplama:** FieldPlant veya başka saha veri setlerinden PlantDoc sınıflarına uyan görüntüleri finetune pool'a eklemek. Bu, diğer tüm yaklaşımların önünde geliyor.

**2. Finetune split değiştirme:** %20→%30 finetune oranı, bacterial_spot ve mosaic_virus için finetune pool'u büyütür. Ancak test set küçülür — değerlendirme güvenilirliği düşer.

**3. Sınıf bazlı augmentation:** bacterial_spot ve mosaic_virus için daha agresif augmentation. Veri yoksa augmentation çok yardımcı olmaz ama denenebilir.

**Artık denenmeyecekler:** AdaBN (trainable backbone ile zararlı), WeightedRandomSampler+FocalLoss (double-counting), daha derin KAT mimarisi (mimari bottleneck değil).

**Önemli:** Her deneme `run_cv_v2.py` ile CV'den geçmeli — test seti epoch seçiminde açılmayacak.

---

## Attention Map Analizi

`visualize_kat_attention.py` KATv2 için güncellendi (16 agent, 4-head → head ortalaması, 4×4 grid). `kat_v2_plantdoc_full_best.pth` checkpoint'i ile PlantDoc test setinden her sınıf için 1 görüntü işlendi. Çıktılar: `results/figures/kat_v2_attention/`.

**8 örnek üzerinde tahmin sonuçları:** healthy ✓, late_blight ✓, mosaic_virus ✓, septoria ✓ — bacterial_spot ✗ (→ septoria), early_blight ✗ (→ yellow_virus), mold ✗ (→ healthy), yellow_virus ✗ (→ late_blight).

**Tespit edilen iki hata türü:**

1. **Görsel benzerlik karışıklığı:** bacterial_spot → septoria_leaf_spot. Her ikisi de küçük koyu leke morfolojisi; model bu iki sınıfı ayırt edemiyor. Attention haritalarında agent'lar leke bölgelerine odaklanıyor ama hangi leke tipinin hangi sınıfa ait olduğunu öğrenememiş.

2. **Hastalık evresi farklılığı:** early_blight → yellow_virus. early_blight ileri evrede sarımsı doku bozulması gösteriyor; model bunu yellow_virus'un mozaik sarısıyla karıştırıyor. Evre bazlı görsel örtüşme, saha fotoğraflarında daha belirgin.

**Mold watermark sorunu:** Bazı mold görüntülerinde veri seti watermark'ı mevcut. Agent'ların bir kısmı watermark bölgesine odaklanıyor — bu gürültülü bir öğrenme sinyali. mold recall=0.22'nin bir kısmı bu sebepten kaynaklanıyor olabilir.

**Genel gözlem:** late_blight ve septoria'da agent'lar hastalık bölgelerine net odaklanıyor. Düşük F1'li sınıflarda (bacterial_spot, early_blight) agent'lar dağınık veya arka plana odaklanıyor.

---

## Çalışma Kuralları

Her değişiklik tek bir amaca hizmet eder. Birden fazla dosyayı aynı anda değiştirmek yok. Bir şeyi değiştirmeden önce neden değiştireceğini açıkla, onay al.

Her çalışan değişiklik bir commit. Mesaj formatı: `feat:` yeni özellik, `fix:` hata düzeltme, `exp:` deney, `docs:` dokümantasyon, `chore:` temizlik. Push'u unutma.

`config.py` dokunulmaz bölge. Değiştirmen gerekiyorsa önce söyle, neden gerektiğini açıkla.

MPS uyumluluğu şart. CUDA'ya özel operasyonlar kullanma — cihaz M2 Air.

Her oturum sonunda CLAUDE.md'yi güncelle ve commit at. Bir sonraki oturumda buradan başlıyoruz, eksik bilgi bırakma.

Büyük fikirlere atlamak yok. Her adımı birlikte değerlendiriyoruz. Saçmalamaya başlarsan kullanıcı müdahale edecek.

---

## Commit Geçmişi

| Tarih    | Commit                                   | Açıklama                                              |
|----------|------------------------------------------|-------------------------------------------------------|
| 9 Mayıs  | init                                     | Temiz proje kurulumu                                  |
| 9 Mayıs  | fix: freeze_backbone                     | Classifier açık kalacak şekilde düzeltildi            |
| 9 Mayıs  | fix: compute_class_weights               | Sıfır bölme koruması eklendi                          |
| 9 Mayıs  | fix: fewshot_finetune validation         | Train/val split, early stopping val loss'a göre       |
| 9 Mayıs  | feat: add KATModel                       | Prototype attention + MLP, smoke-test geçti           |
| 9 Mayıs  | feat: add train_kat.py                   | KAT eğitim pipeline                                   |
| 9 Mayıs  | exp: evaluate KAT baseline               | PlantDoc acc 0.2341 balanced 0.2173                   |
| 9 Mayıs  | fix: revert to frozen backbone           | Sadece agent_queries + classifier eğitilir            |
| 9 Mayıs  | feat: add init_kat_prototypes.py         | PlantDoc class prototype vektörleri                   |
| 9 Mayıs  | feat: fewshot_kat — prototype init       | agent_queries PlantDoc prototipleriyle başlatılıyor   |
| 9 Mayıs  | feat: add finetune_full()                | Tüm finetune pool ile KAT eğitimi                     |
| 9 Mayıs  | exp: differential lr in finetune_full    | Backbone 1e-5, head 1e-4                              |
| 9 Mayıs  | chore: remove unused files               | Smoke scripts ve MMD pipeline temizlendi              |
| 9 Mayıs  | feat: add KATModelV2                     | 16 agent, 4-head attention, 14×14 feature map         |
| 9 Mayıs  | feat: add train_kat_v2.py                | KATv2 PlantVillage pre-training (val acc 0.9869)      |
| 9 Mayıs  | feat: add fewshot_kat_v2.py              | KATv2 full finetune pipeline                          |
| 9 Mayıs  | exp: KATv2 full finetune                 | No val split, fixed epochs — acc 0.4130 @ epoch 40   |
| 10 Mayıs | exp: add diversity loss to KATv2         | Balanced acc 0.4290 @ ep50, 0.4510 @ ep60             |
| 10 Mayıs | docs: update CLAUDE.md                   | Diversity loss sonuçları, metodoloji uyarısı, CV + AdaBN planı |
| 13 Mayıs | feat: add train_one_fold() + run_cv()    | 5-fold stratified CV; freeze blocks.4 → blocks.6+conv_head |
| 13 Mayıs | feat: add run_cv_v2.py                   | CV runner script                                      |
| 13 Mayıs | feat: add apply_adabn() + evaluate_with_adabn() | num_epochs parametrik; AdaBN fonksiyonları   |
| 13 Mayıs | feat: add run_adabn_v2.py                | AdaBN runner (checkpoint argümanı alır)               |
| 13 Mayıs | docs: update CLAUDE.md                   | CV sonuçları, AdaBN analizi, sınıf bazlı breakdown    |
| 15 Mayıs | feat: FocalLoss (gamma=2) — train_fewshot + train_one_fold | CV avg 0.3729→0.4025, test 0.3545→0.3575 |
| 15 Mayıs | feat: update visualize_kat_attention.py for KATv2 | 16 agent, 4×4 grid, head-averaged attn  |
| 15 Mayıs | docs: update CLAUDE.md                   | Focal Loss + attention analizi, oturum sonu 15 Mayıs  |
| 15 Mayıs | feat: add run_cv_efficientnet.py          | EfficientNet 5-fold CV + full retrain; best ep=20, balanced=0.3553 |
| 15 Mayıs | docs: update CLAUDE.md                   | EfficientNet CV sonuçları, mimari karşılaştırma, veri darboğazı tespiti |

---

## Bağımlılıklar

```
torch, timm, albumentations, scikit-learn, matplotlib, seaborn, Pillow, numpy
```
