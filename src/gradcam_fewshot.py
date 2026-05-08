import os
import config
from dataset import load_plantdoc_samples, split_plantdoc, get_few_shot_samples
from gradcam_utils import (
    load_model,
    predict_dataset,
    select_samples_per_class,
    generate_gradcam_figure
)


def main():
    all_samples = load_plantdoc_samples(config.PLANTDOC_DIR)
    finetune_pool, test_samples = split_plantdoc(all_samples)
    print(f"Test set: {len(test_samples)} images")

    for n_shot in config.FEW_SHOT_COUNTS:
        model_path = os.path.join(config.MODELS_DIR,
                                  f"plantdoc_{n_shot}shot_best.pth")
        save_path  = os.path.join(config.RESULTS_DIR, "figures",
                                  f"gradcam_{n_shot}shot.png")

        if not os.path.exists(model_path):
            print(f"Model not found, skipping: {model_path}")
            continue

        print(f"\nLoading {n_shot}-shot model...")
        model, device = load_model(model_path)

        print("Running predictions...")
        correct_samples, wrong_samples = predict_dataset(model, device, test_samples)
        print(f"Correct: {len(correct_samples)} | Wrong: {len(wrong_samples)}")

        selected_correct, selected_wrong = select_samples_per_class(
            correct_samples, wrong_samples, n_per_class=1
        )

        generate_gradcam_figure(
            model, device,
            selected_correct, selected_wrong,
            title=f"GradCAM Analysis — {n_shot}-Shot Fine-tuned Model (PlantDoc)",
            save_path=save_path
        )


if __name__ == "__main__":
    main()