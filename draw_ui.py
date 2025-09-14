import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas 

st.set_page_config(page_title="Digit Recognizer")
st.title("Draw a Digit")

model = load_model("models/digit_cnn_model.h5")

canvas = canvas = st_canvas(
    width=280,
    height=280,
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    drawing_mode="freedraw",
    key="canvas"
)

if canvas.image_data is not None:
    img = canvas.image_data
    img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)  # Invert black/white
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    if st.button("Predict"):
        pred = model.predict(img)
        st.write(f"### Predicted Digit: {np.argmax(pred)}")