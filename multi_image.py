import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# ------------------------------
# Load Model
# ------------------------------
MODEL_PATH = "models/efficientnet_fish.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Fish classes
CLASSES = ["Salmon", "Tuna", "Trout", "Mackerel", "Sardine"]

# ------------------------------
# Function to Process Images
# ------------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🐟 Fish Image Classification App")
st.subheader("Upload a fish image and let the model predict the species.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    # Display uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # Prediction
    prediction = model.predict(img_array)[0]

    # Get highest score
    pred_class = CLASSES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🎯 **Prediction: {pred_class}**")
    st.info(f"🔎 Confidence: {confidence:.2f}%")

    # Show confidence scores for all classes
    st.subheader("Confidence Scores:")
    for i, cls in enumerate(CLASSES):
        st.write(f"**{cls}:** {prediction[i]*100:.2f}%")
