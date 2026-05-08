import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import config
from model import build_model, get_device
from dataset import get_val_transform


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        score = output[0, class_idx]
        score.backward()

        # Global average pooling ile kanal agirliklarini hesapla
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Orijinal goruntu boyutuna yeniden olcekle
        cam = F.interpolate(cam, size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # 0-1 araligina normalize et
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, output.softmax(dim=1)[0, class_idx].item()


def get_target_layer(model):
    # EfficientNet-B0'da son konvolusyon blogu
    return model.blocks[-1][-1].bn3


def load_model(model_path):
    device = get_device()
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device


def predict_dataset(model, device, samples):
    """
    Verilen ornekler uzerinde tahmin yapar.
    Doğru ve yanliş tahminleri ayri listeler olarak dondurur.
    """
    transform = get_val_transform()
    correct_samples = []
    wrong_samples   = []

    for img_path, true_label in samples:
        image_np = np.array(Image.open(img_path).convert("RGB"))
        tensor   = transform(image=image_np)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            output    = model(tensor)
            pred_label = output.argmax(dim=1).item()

        entry = (img_path, true_label, pred_label)
        if pred_label == true_label:
            correct_samples.append(entry)
        else:
            wrong_samples.append(entry)

    return correct_samples, wrong_samples


def select_samples_per_class(correct_samples, wrong_samples, n_per_class=1, seed=config.SEED):
    """
    Her siniftan n_per_class dogru ve n_per_class yanlis ornek secer.
    Ornekler rastgele secilir ama seed ile tekrarlanabilir.
    """
    import random
    random.seed(seed)

    correct_by_class = {}
    for item in correct_samples:
        correct_by_class.setdefault(item[1], []).append(item)

    wrong_by_class = {}
    for item in wrong_samples:
        wrong_by_class.setdefault(item[1], []).append(item)

    selected_correct = []
    selected_wrong   = []

    for class_idx in range(config.NUM_CLASSES):
        c_items = correct_by_class.get(class_idx, [])
        w_items = wrong_by_class.get(class_idx, [])

        if c_items:
            selected_correct.append(random.choice(c_items))
        if w_items:
            selected_wrong.append(random.choice(w_items))

    return selected_correct, selected_wrong


def overlay_cam_on_image(image_np, cam, alpha=0.5):
    """
    GradCAM isi haritasini orijinal goruntu uzerine bindirer.
    """
    heatmap = plt.cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    image_resized = np.array(
        Image.fromarray(image_np).resize(
            (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR
        )
    )
    overlay = (alpha * heatmap + (1 - alpha) * image_resized).astype(np.uint8)
    return overlay


def generate_gradcam_figure(model, device, selected_correct, selected_wrong,
                             title, save_path):
    """
    Dogru ve yanlis tahminler icin GradCAM gorseli olusturur.
    Her satir bir sinif, sol sutun dogru tahmin, sag sutun yanlis tahmin.
    """
    target_layer = get_target_layer(model)
    gradcam      = GradCAM(model, target_layer)
    transform    = get_val_transform()

    # Sinif listesini olustur - her ikisinde de olan siniflar
    correct_classes = {item[1] for item in selected_correct}
    wrong_classes   = {item[1] for item in selected_wrong}
    all_classes     = sorted(correct_classes | wrong_classes)

    correct_map = {item[1]: item for item in selected_correct}
    wrong_map   = {item[1]: item for item in selected_wrong}

    n_rows = len(all_classes)
    n_cols = 4  # orijinal + cam (dogru) | orijinal + cam (yanlis)

    fig = plt.figure(figsize=(16, n_rows * 3.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    # Sutun basliklari
    col_titles = [
        "Original (Correct)",
        "GradCAM (Correct)",
        "Original (Wrong)",
        "GradCAM (Wrong)"
    ]

    outer = gridspec.GridSpec(n_rows, 1, hspace=0.6)

    for row_idx, class_idx in enumerate(all_classes):
        class_name = config.CLASS_NAMES[class_idx].replace("_", " ").title()
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer[row_idx], wspace=0.05
        )

        for col_idx in range(n_cols):
            ax = fig.add_subplot(inner[col_idx])
            ax.axis("off")

            # Sutun basligi sadece ilk satirda
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=9, pad=4)

            is_correct_side = col_idx < 2
            item_map        = correct_map if is_correct_side else wrong_map
            item            = item_map.get(class_idx, None)

            if item is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                continue

            img_path, true_label, pred_label = item
            image_np = np.array(Image.open(img_path).convert("RGB"))
            tensor   = transform(image=image_np)["image"].unsqueeze(0).to(device)

            if col_idx % 2 == 0:
                # Orijinal goruntu
                display = np.array(
                    Image.fromarray(image_np).resize(
                        (config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR
                    )
                )
                ax.imshow(display)

                # Dogru/yanlis etiket
                label_color = "#2ecc71" if is_correct_side else "#e74c3c"
                status      = "Correct" if is_correct_side else "Wrong"
                pred_name   = config.CLASS_NAMES[pred_label].replace("_", " ").title()
                ax.set_xlabel(
                    f"{status}\nPred: {pred_name}",
                    fontsize=7, color=label_color, labelpad=2
                )
            else:
                # GradCAM overlay
                cam, _, confidence = gradcam.generate(tensor, class_idx=true_label)
                overlay = overlay_cam_on_image(image_np, cam)
                ax.imshow(overlay)
                ax.set_xlabel(
                    f"Conf: {confidence:.2f}",
                    fontsize=7, color="gray", labelpad=2
                )

        # Satir etiketi - sinif adi
        fig.text(
            0.01,
            1 - (row_idx + 0.5) / n_rows,
            class_name,
            va="center", ha="left",
            fontsize=8, fontweight="bold",
            rotation=90
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")