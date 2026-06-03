import sys
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import BASE_DIR, IMAGE_SIZE, CLASS_NAMES


MODEL_PATH = BASE_DIR / "saved_models" / "final_mobilenetv2_180" / "best_model.keras"


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict_final path/to/image.jpg")
        return

    image_path = sys.argv[1]

    model = tf.keras.models.load_model(MODEL_PATH)

    img_array = load_image(image_path)

    preds = model.predict(img_array, verbose=0)[0]

    pred_index = np.argmax(preds)
    confidence = preds[pred_index] * 100

    print("Image:", image_path)
    print("Predicted class:", CLASS_NAMES[pred_index])
    print(f"Confidence: {confidence:.2f}%")

    print("\nAll class probabilities:")
    for class_name, prob in zip(CLASS_NAMES, preds):
        print(f"{class_name}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()