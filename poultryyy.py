import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Load class names
class_names = open("class_names.txt").read().splitlines()

st.title("🐔 Poultry Health Monitoring System")
st.write("Upload a poultry droppings image to detect disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("Prediction:")
    st.success(f"{predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

with open("class_names.txt") as f:
    class_names = f.read().splitlines()


