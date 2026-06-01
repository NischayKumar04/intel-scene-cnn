import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.dataset import get_datasets
from src.config import SAVED_MODEL_DIR, FIGURES_DIR


def main():
    _, _, test_ds, class_names = get_datasets()

    model = tf.keras.models.load_model(SAVED_MODEL_DIR / "best_custom_cnn_bn_after_relu_label_smoothing_l2_180.keras")

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

    plt.title("Confusion Matrix - Custom CNN")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_custom_cnn.png")
    plt.show()


if __name__ == "__main__":
    main()