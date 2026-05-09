# CLAUDE.md — Cross-Domain Plant Disease Classification

**Son güncelleme:** 10 Mayıs 2026

---

## Proje Hakkında

Bu proje, domates yaprak hastalıklarını kontrollü laboratuvar ortamından (PlantVillage) saha fotoğraflarına (PlantDoc) taşıma problemini ele alıyor. Temel zorluk şu: lab ortamında %99'un üzerinde doğrulukla çalışan bir model, saha koşullarında dramatik biçimde başarısız oluyor. Bunu domain gap problemi olarak adlandırıyoruz.

KATv1 deneyleri tamamlandı; EfficientNet few-shot'ın gerisinde kaldı. KATv2 (16 agent, 4-head attention, 14×14 feature map) ile val split kaldırılıp sabit epoch eğitimine geçildi. Diversity loss (lambda=0.01, cosine sim cezası) eklenerek epoch 60'ta **acc 0.4565 / balanced 0.4510** elde edildi. Sıradaki: 5-fold cross-validation + AdaBN.

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
│   ├── run_finetune_full_v2.py    # KATv2 finetune_full() çağırır
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
| **KATv2 Full + div loss — ep60** | **0.4565** | **0.4510** | Tamamlandı |

Hedef: balanced accuracy > 0.60

**Önemli:** KATv2 epoch 40'ta EfficientNet 10-shot'ı (0.3880 / 0.3989) accuracy'de geçti. Epoch 60'ta balanced accuracy 0.4510'a ulaştı; train_loss=1.2945, trend hâlâ iniyor.

Val split kaldırıldı — tüm 146 finetune örneği direkt train'e veriliyor, sabit epoch sayısı kullanılıyor.

### ⚠️ Metodoloji Uyarısı

**Şu an test setine bakarak epoch seçiyoruz — bu data leakage'dır.** Her 10 epoch'ta test sonucuna bakıp "daha fazla eğitelim" kararı vermek, modeli dolaylı olarak test setine göre seçmek anlamına gelir. Bu durum, raporlanan metriklerin gerçek genelleme performansından iyimser olmasına yol açar.

Doğru yaklaşım: test setine hiç bakmadan epoch sayısına karar vermek. Bunu yapmanın tek güvenilir yolu 5-fold cross-validation — fold'ların ortalaması erken durma / epoch seçimi için kullanılır, test seti yalnızca final değerlendirmede açılır.

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
EfficientNet-B0 (backbone — blocks.4 kısmen açık, geri kalan donuk)
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
- Sabit epoch sayısı, her epoch sonunda checkpoint kaydedilir
- Differential lr: backbone blocks.4 → 1e-5, head → 1e-4
- Her 10 epoch'ta test set üzerinde ara değerlendirme yapılır

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
- Epoch 60'ta acc 0.4565 / balanced 0.4510; train_loss=1.2945 — loss hâlâ iniyor, plateau yok.
- **Metodoloji sorunu:** test setine bakarak epoch seçildi → 5-fold CV ile düzeltilmeli.

---

## Sıradaki Adımlar

### a) 5-Fold Cross-Validation

146 finetune örneğini 5 fold'a böl. Her fold için: 4 fold'u train'e ver, 1 fold'u val'e ayır, sabit epoch sayısıyla eğit, val metriğini kaydet. 5 fold ortalaması epoch sayısı kararı için kullanılır — test seti bu aşamada hiç açılmaz. Sonra seçilen epoch sayısıyla tüm 146 örnek üzerinde yeniden eğit, yalnızca o zaman test setini aç.

### b) AdaBN (Adaptive Batch Normalization)

Inference-time domain adaptation — eğitim kodu değişmez, sadece değerlendirme adımına eklenir.

**Nasıl çalışır:**
1. Model checkpoint yüklendikten sonra `model.train()` moduna al.
2. PlantDoc finetune görüntüleriyle (veya test görüntüleriyle) birkaç forward pass yap — gradient hesaplanmaz (`torch.no_grad()`), sadece BN running stats (running_mean, running_var) güncellenir.
3. `model.eval()` moduna geç, test setini değerlendir.

**Amaç:** Backbone'daki BN katmanları PlantVillage istatistikleriyle dolu; PlantDoc görüntülerini göstererek bu istatistikleri saha dağılımına yakınsatmak.

**Not:** KATv2'de backbone büyük ölçüde dondurulmuş olsa da blocks[4] açık — o bloktaki BN istatistikleri güncellenecek.

---

## Attention Map Analizi

`visualize_kat_attention.py` ile PlantDoc test setindeki örneklerin agent attention haritaları incelendi:

- Class 7 (Septoria) ve Class 3 (Late Blight) en iyi sonuç — agent'lar hastalık bölgelerine net odaklanıyor.
- Class 4 (Leaf Miner) veri kalitesi sorunu — watermark mevcut, model watermark'ı öğreniyor.
- Class 1 ve 6'da birden fazla yaprak aynı karede; agent'lar hangi yaprağa bakacağını seçemiyor.
- Class 2 ve 5'te backbone PlantVillage'e kilitli; agent'lar arka plana odaklanıyor.

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

---

## Bağımlılıklar

```
torch, timm, albumentations, scikit-learn, matplotlib, seaborn, Pillow, numpy
```
