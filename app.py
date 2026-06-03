import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.config import BASE_DIR, IMAGE_SIZE, CLASS_NAMES


MODEL_PATH = BASE_DIR / "saved_models" / "final_mobilenetv2_180" / "best_model.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


st.set_page_config(
    page_title="Intel Scene Classifier",
    page_icon="🌄",
    layout="centered",
)

st.title("Intel Scene Classification")
st.write("Upload an image and the model will classify it into one of six scene classes.")

st.markdown(
    """
    **Classes:** buildings, forest, glacier, mountain, sea, street  
    **Model:** Fine-tuned MobileNetV2  
    **Input size:** 180 × 180  
    **Test accuracy:** ~92.47%
    """
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()

    input_array = preprocess_image(image)

    preds = model.predict(input_array, verbose=0)[0]

    pred_index = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_index]
    confidence = float(preds[pred_index]) * 100

    st.subheader("Prediction")
    st.success(f"{pred_class} ({confidence:.2f}%)")

    st.subheader("Class Probabilities")

    probs = {
        class_name: float(prob) * 100
        for class_name, prob in zip(CLASS_NAMES, preds)
    }

    st.bar_chart(probs)

    st.write("Raw probabilities:")
    st.json({k: f"{v:.2f}%" for k, v in probs.items()})
else:
    st.info("Upload an image to get a prediction.")