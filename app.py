import os
import cv2
import urllib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import streamlit as st

# Load model
model = model_from_json(open("model.json", "r").read())
model.load_weights('weights_model1.h5')

# Loading the classifier from the file.
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit app
st.title('Face Expression Recognition')

# Image upload section
st.header('Upload an Image')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        predicted_emotion = emotions[np.argmax(predictions)]

        st.image(img, caption=f"Predicted Emotion: {predicted_emotion}", use_column_width=True)

# Real-time video section
st.header('Real-time Video Analysis')

# You can use OpenCV and VideoCapture as before to capture video frames and display them in Streamlit.

# The rest of your Flask code for URL image upload can be similarly adapted to Streamlit.
