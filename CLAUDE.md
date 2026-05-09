# CLAUDE.md — Cross-Domain Plant Disease Classification

**Son güncelleme:** 9 Mayıs 2026

---

## Proje Hakkında

Bu proje, domates yaprak hastalıklarını kontrollü laboratuvar ortamından (PlantVillage) saha fotoğraflarına (PlantDoc) taşıma problemini ele alıyor. Temel zorluk şu: lab ortamında %99'un üzerinde doğrulukla çalışan bir model, saha koşullarında dramatik biçimde başarısız oluyor. Bunu domain gap problemi olarak adlandırıyoruz.

Mevcut few-shot fine-tuning yaklaşımı bu boşluğu kapatmak için yetersiz kalıyor — en iyi sonuç 10-shot ile %39 balanced accuracy. Hedef, Kernel-guided Agent Transformer (KAT) mimarisini kullanarak bu oranı %60'ın üzerine çıkarmak.

KAT'ın temel fikri şu: her agent, feature map üzerinde farklı bir görsel örüntüye odaklanıyor — leke, renk bozulması, doku değişimi gibi. Bu sayede model, saha görüntülerindeki değişken koşullara rağmen hastalık belirtilerini doğru bölgelerden okuyabiliyor. Agent çıktıları MLP ile birleştirilerek sınıf kararı veriliyor.

---

## Proje Yapısı

```
plant_disease_project/
├── src/
│   ├── config.py               # Tüm sabitler burada — değiştirmeden önce mutlaka sor
│   ├── dataset.py              # Veri yükleme, augmentation, few-shot sampling
│   ├── model.py                # EfficientNet-B0, freeze/unfreeze yardımcıları
│   ├── model_kat.py            # KAT modeli — bu oturumda tamamlandı
│   ├── train_kat.py            # KAT eğitim pipeline
│   ├── train.py                # PlantVillage üzerinde baseline eğitimi
│   ├── fewshot_finetune.py     # PlantDoc few-shot fine-tuning (5 ve 10-shot)
│   ├── evaluate.py             # Tüm modellerin değerlendirilmesi
│   ├── gradcam_utils.py        # GradCAM altyapısı ve görselleştirme
│   ├── gradcam_baseline.py     # Baseline model için GradCAM
│   └── gradcam_fewshot.py      # Few-shot modeller için GradCAM
├── scripts/
│   ├── smoke_kat.py            # KATModel forward pass testi
│   ├── run_smoke_train_kat.py  # temporary smoke-run wrapper for train_kat
│   └── evaluate_kat_baseline.py# evaluate saved KAT model on PlantDoc test set
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

| Model                 | Accuracy | Balanced Acc | Durum        |
|-----------------------|----------|--------------|--------------|
| EfficientNet Baseline | 0.2943   | 0.2517       | Tamamlandı   |
| EfficientNet 5-shot   | 0.3763   | 0.3770       | Tamamlandı   |
| EfficientNet 10-shot  | 0.3880   | 0.3989       | Tamamlandı   |
| KAT Baseline          | 0.2341   | 0.2173       | Tamamlandı   |
| KAT 5-shot            | —        | —            | Sırada       |
| KAT 10-shot           | —        | —            | Sırada       |

Hedef: KAT 10-shot balanced accuracy > 0.60

---

## KAT Mimarisi

Model şu akışı izliyor:

```
EfficientNet-B0 (dondurulmuş backbone)
    → forward_features → (B, 1280, 7, 7)
    → 1x1 Conv projection → (B, 256, 7, 7)
    → spatial flatten → (B, 49, 256)
    → Cross-attention: 8 agent query × 49 spatial token
    → agent outputs → (B, 8, 256)
    → LayerNorm → Flatten → (B, 2048)
    → MLP: 2048 → 512 → 256 → NUM_CLASSES
```

Her agent farklı bir görsel örüntüye odaklanıyor. Hangi bölgeye ne kadar dikkat ettiği, attention map olarak geri dönebiliyor — bu da yorumlanabilirlik açısından önemli.

Config'deki KAT sabitleri:

```python
KAT_NUM_AGENTS  = 8
KAT_AGENT_DIM   = 256
KAT_MLP_HIDDEN  = [512, 256]
KAT_DROPOUT     = 0.3
```

Smoke-test sonucu (8 Mayıs 2026):
- logits shape: (1, 8) — doğru
- attention maps shape: (1, 8, 7, 7) — doğru
- Eğitilebilir parametre sayısı: 1,512,968

---

## Sıradaki Adımlar

Öncelik sırasıyla:

1. `src/fewshot_kat.py` — sadece agent_queries'i fine-tune et, backbone donuk kalır
2. `src/evaluate_kat.py` — KAT sonuçlarını baseline tablosuyla karşılaştır (genel değerlendirme)
3. KAT prototipleri iyileştirme ve MMD aşamaları (deneysel)

---

## Çalışma Kuralları

Bu proje boyunca şu kurallara uyuyoruz:

Her değişiklik tek bir amaca hizmet eder. Birden fazla dosyayı aynı anda değiştirmek yok. Bir şeyi değiştirmeden önce neden değiştireceğini açıkla, onay al.

Her çalışan değişiklik bir commit. Mesaj formatı: `feat:` yeni özellik, `fix:` hata düzeltme, `exp:` deney. Push'u unutma.

`config.py` dokunulmaz bölge. Değiştirmen gerekiyorsa önce söyle, neden gerektiğini açıkla.

MPS uyumluluğu şart. CUDA'ya özel operasyonlar kullanma — cihaz M2 Air.

Her oturum sonunda CLAUDE.md'yi güncelle ve commit at. Bir sonraki oturumda buradan başlıyoruz, eksik bilgi bırakma.

Büyük fikirlere atlamak yok. Her adımı birlikte değerlendiriyoruz. Saçmalamaya başlarsan kullanıcı müdahale edecek.

---

## Commit Geçmişi

| Tarih       | Commit                              | Açıklama                                        |
|-------------|-------------------------------------|-------------------------------------------------|
| 9 Mayıs     | init                                | Temiz proje kurulumu                            |
| 9 Mayıs     | fix: freeze_backbone                | Classifier açık kalacak şekilde düzeltildi      |
| 9 Mayıs     | fix: compute_class_weights          | Sıfır bölme koruması eklendi                    |
| 9 Mayıs     | fix: fewshot_finetune validation    | Train/val split, early stopping val loss'a göre |
| 9 Mayıs     | feat: add KATModel                  | Prototype attention + MLP, smoke-test geçti     |
| 9 Mayıs     | feat: add train_kat.py              | KAT eğitim pipeline                              |
| 9 Mayıs     | exp: evaluate KAT baseline          | PlantDoc acc 0.2341 balanced 0.2173              |

---

## Bağımlılıklar

```
torch, timm, albumentations, scikit-learn, matplotlib, seaborn, Pillow, numpy
```
