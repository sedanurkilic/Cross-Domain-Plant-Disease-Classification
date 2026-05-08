import timm
import torch
import torch.nn as nn
import config

def gaussian_kernel(x, y, kernel_num=5, kernel_mul=2.0):
    n_samples = x.shape[0] + y.shape[0]
    total     = torch.cat([x, y], dim=0)
    total_sq  = (total ** 2).sum(dim=1, keepdim=True)
    dist      = total_sq + total_sq.t() - 2 * torch.mm(total, total.t())
    dist      = dist.clamp(min=0)
    bandwidth = dist.sum() / (n_samples ** 2 - n_samples + 1e-6)
    bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernels   = sum(
        torch.exp(-dist / (bandwidth * (kernel_mul ** i)))
        for i in range(kernel_num)
    )
    return kernels


def mmd_loss(source, target, kernel_num=5):
    n        = source.shape[0]
    m        = target.shape[0]
    kernels  = gaussian_kernel(source, target, kernel_num=kernel_num)
    xx       = kernels[:n, :n].mean()
    yy       = kernels[n:, n:].mean()
    xy       = kernels[:n, n:].mean()
    return xx + yy - 2 * xy

def build_model(num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED):
    model = timm.create_model(
        config.BACKBONE,
        pretrained=pretrained,
        num_classes=0
    )

    in_features = model.num_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=config.DROPOUT_RATE),
        nn.Linear(in_features, num_classes)
    )

    return model


def freeze_backbone(model):
    """
    Few-shot fine-tuning icin:
    Backbone tamamen dondurulur, sadece classifier acik kalir.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


def freeze_all_except_classifier(model):
    """
    Stage 1 MMD adaptasyonu icin backbone tamamen dondurulur.
    BN katmanlari model.train() modunda running stats guncellemeye
    devam eder — AdaBN etkisi saglar.
    Sadece classifier guncellenmez, o da dondurulur.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_for_finetuning(model):
    for param in model.parameters():
        param.requires_grad = False

    # blocks[5] ve blocks[6] ac
    for name, param in model.named_parameters():
        if "blocks.5" in name or "blocks.6" in name or "conv_head" in name:
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    print("Unfrozen: blocks[5] + blocks[6] + conv_head + classifier")


def get_domain_features(model, x):
    """
    MMD hesabi icin global average pooled feature vektor dondurur.
    EfficientNet'in conv_head cikisini kullanir: (B, 1280)
    Classifier'dan once, backbone cikisi.
    """
    features = model.forward_features(x)
    # EfficientNet timm modelinde forward_features (B, 1280, 7, 7) dondurur
    # Global average pooling ile (B, 1280) yapiyoruz
    if features.dim() == 4:
        features = features.mean(dim=(2, 3))
    return features


def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")