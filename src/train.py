import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf
import matplotlib.pyplot as plt

from src.dataset import get_datasets
from src.model import build_custom_cnn_bn_after_relu_l2 as build_custom_cnn
from src.config import EPOCHS, SAVED_MODEL_DIR, FIGURES_DIR


EXPERIMENT_NAME = "bn_after_relu_label_smoothing_l2_180"


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU detected:", gpus)
else:
    print("No GPU detected. Training will use CPU.")


try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled.")
except Exception as e:
    print("Mixed precision not enabled:", e)


def plot_training_curves(history, experiment_name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve - {experiment_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{experiment_name}_accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {experiment_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{experiment_name}_loss_curve.png")
    plt.close()


def main():
    train_ds, val_ds, test_ds, class_names = get_datasets()

    print("Classes:", class_names)

    model = build_custom_cnn(num_classes=len(class_names))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    model.summary()

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(SAVED_MODEL_DIR / f"best_custom_cnn_{EXPERIMENT_NAME}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_training_curves(history, EXPERIMENT_NAME)

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    model.save(SAVED_MODEL_DIR / f"final_custom_cnn_{EXPERIMENT_NAME}.keras")


if __name__ == "__main__":
    main()