import os

# Proje kök dizini 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Veri yolları
PLANTVILLAGE_DIR = os.path.join(BASE_DIR, "data", "processed", "tomato_plantvillage")
PLANTDOC_DIR     = os.path.join(BASE_DIR, "data", "processed", "tomato_plantdoc")
MODELS_DIR       = os.path.join(BASE_DIR, "models")
RESULTS_DIR      = os.path.join(BASE_DIR, "results")

# PlantVillage ve PlantDoc klasör isimlerinin eşleşmesi
# Sol taraf: PlantVillage klasör adı, Sağ taraf: PlantDoc klasör adı
CLASS_MAPPING = {
    "tomato_leaf_bacterial_spot"       : "tomato_leaf_bacterial_spot",
    "tomato_leaf_early_blight"         : "tomato_leaf_early_blight",
    "tomato_leaf_late_blight"          : "tomato_leaf_late_blight",
    "tomato_leaf_mold"                 : "tomato_leaf_mold",
    "tomato_leaf_mosaic_virus"         : "tomato_leaf_mosaic_virus",
    "tomato_leaf_yellow_virus"         : "tomato_leaf_yellow_virus",
    "tomato_leaf_healthy"              : "tomato_leaf_healthy",
    "tomato_septoria_leaf_spot"        : "tomato_septoria_leaf_spot",
}

# Sınıf indeksleri ve isimleri
CLASS_NAMES = sorted(CLASS_MAPPING.keys())
NUM_CLASSES = len(CLASS_NAMES)  # 8

# Model
BACKBONE        = "efficientnet_b0"
PRETRAINED      = True
IMAGE_SIZE      = 224
DROPOUT_RATE    = 0.3

# PlantVillage eğitimi
TRAIN_VAL_SPLIT = 0.8       # %80 train, %20 val
BATCH_SIZE      = 8
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
EARLY_STOPPING  = 7         # val loss 7 epoch iyileşmezse dur

# KAT v3 - Unsupervised Domain Adaptation
MMD_LAMBDA      = 0.5       # MMD loss agirligi
MMD_KERNEL_NUM  = 5         # RBF kernel sayisi
UDA_BATCH_SIZE  = 16        # PlantDoc referans batch boyutu
FEATURE_DIM     = 512      # EfficientNet 1280

# KAT (Kernel-guided Agent Transformers) - lightweight tunables
KAT_NUM_AGENTS = 8
KAT_AGENT_DIM  = 256
KAT_MLP_HIDDEN = [512, 256]
KAT_DROPOUT    = 0.3

# Few-shot fine-tuning
FEW_SHOT_COUNTS = [5, 10]   # her ikisini de deneceğiz
FINETUNE_EPOCHS = 30
FINETUNE_LR     = 5e-3       # backbone donuk olduğu için düşük tutuyoruz

# PlantDoc split oranları
PLANTDOC_TEST_RATIO     = 0.8   # %80 test, hiç dokunulmaz
PLANTDOC_FINETUNE_RATIO = 0.2   # %20 few-shot fine-tuning için

# Tekrarlanabilirlik
SEED = 42

# Hesaplama
NUM_WORKERS = 0   