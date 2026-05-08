import os
import config
from dataset import load_plantdoc_samples, split_plantdoc
from gradcam_utils import (
    load_model,
    predict_dataset,
    select_samples_per_class,
    generate_gradcam_figure
)


def main():
    model_path = os.path.join(config.MODELS_DIR, "plantvillage_best.pth")
    save_path  = os.path.join(config.RESULTS_DIR, "figures",
                              "gradcam_baseline.png")

    print("Loading baseline model (no fine-tuning)...")
    model, device = load_model(model_path)

    # Test setini yukle - ayni seed ile her zaman ayni bolunme
    all_samples = load_plantdoc_samples(config.PLANTDOC_DIR)
    _, test_samples = split_plantdoc(all_samples)
    print(f"Test set: {len(test_samples)} images")

    print("Running predictions...")
    correct_samples, wrong_samples = predict_dataset(model, device, test_samples)
    print(f"Correct: {len(correct_samples)} | Wrong: {len(wrong_samples)}")

    selected_correct, selected_wrong = select_samples_per_class(
        correct_samples, wrong_samples, n_per_class=1
    )

    generate_gradcam_figure(
        model, device,
        selected_correct, selected_wrong,
        title="GradCAM Analysis — Baseline Model (PlantVillage only, no fine-tuning)",
        save_path=save_path
    )


if __name__ == "__main__":
    main()