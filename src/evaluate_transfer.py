import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.dataset import get_datasets
from src.config import SAVED_MODEL_DIR, FIGURES_DIR


EXPERIMENT_NAME = "mobilenetv2_finetune_180_more_layers"
MODEL_NAME = f"best_{EXPERIMENT_NAME}.keras"


def main():
    _, _, test_ds, class_names = get_datasets()

    model = tf.keras.models.load_model(SAVED_MODEL_DIR / MODEL_NAME)

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)

    plt.title(f"Confusion Matrix - {EXPERIMENT_NAME}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{EXPERIMENT_NAME}_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()