# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLO model
@st.cache_resource
def load_model():
    model = YOLO("E:/Sem 5/ML project/5_emotion.pt")  # Load the YOLO model
    return model

model = load_model()

# Define the emotion labels
EMOTIONS = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]

# Streamlit UI
st.title("Facial Emotion Recognition (FER) with YOLO")
st.write("Upload an image to detect facial emotions.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction using YOLO model
    results = model.predict(source=np.array(image), conf=0.25, save=False, device="cpu")

    # Parse the results
    detected_emotions = []
    for result in results[0].boxes.data.tolist():
        _, _, _, _, confidence, class_id = result  # Extract bounding box and class ID
        detected_emotions.append((EMOTIONS[int(class_id)], confidence))

    # Display detected emotions
    if detected_emotions:
        st.write("### Detected Emotions:")
        for emotion, confidence in detected_emotions:
            st.write(f"- {emotion}: {confidence:.2f}")
    else:
        st.write("No emotions detected.")
