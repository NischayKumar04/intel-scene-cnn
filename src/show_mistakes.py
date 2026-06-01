import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.dataset import get_datasets
from src.config import SAVED_MODEL_DIR, FIGURES_DIR


MODEL_NAME = "best_custom_cnn_bn_after_relu_label_smoothing.keras"
EXPERIMENT_NAME = "bn_after_relu_label_smoothing"

# Change these two values to inspect different mistakes
TARGET_TRUE_CLASS = "mountain"
TARGET_PRED_CLASS = "sea"


def main():
    _, _, test_ds, class_names = get_datasets()

    model = tf.keras.models.load_model(SAVED_MODEL_DIR / MODEL_NAME)

    mistakes = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)

        true_ids = np.argmax(labels.numpy(), axis=1)
        pred_ids = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)

        for i in range(len(images)):
            true_class = class_names[true_ids[i]]
            pred_class = class_names[pred_ids[i]]

            if true_class == TARGET_TRUE_CLASS and pred_class == TARGET_PRED_CLASS:
                mistakes.append({
                    "image": images[i].numpy().astype("uint8"),
                    "true": true_class,
                    "pred": pred_class,
                    "confidence": confidences[i],
                })

    print(f"Total {TARGET_TRUE_CLASS} predicted as {TARGET_PRED_CLASS}:", len(mistakes))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = 4
    cols = 4
    total = min(rows * cols, len(mistakes))

    if total == 0:
        print("No mistakes found for this pair.")
        return

    plt.figure(figsize=(14, 12))

    for i in range(total):
        item = mistakes[i]

        plt.subplot(rows, cols, i + 1)
        plt.imshow(item["image"])
        plt.axis("off")
        plt.title(
            f"True: {item['true']}\nPred: {item['pred']}\nConf: {item['confidence']:.2f}",
            fontsize=9,
        )

    plt.tight_layout()

    filename = f"{EXPERIMENT_NAME}_{TARGET_TRUE_CLASS}_as_{TARGET_PRED_CLASS}.png"
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

    print("Saved:", FIGURES_DIR / filename)


if __name__ == "__main__":
    main()