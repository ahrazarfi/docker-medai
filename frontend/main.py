import time

import requests
import streamlit as st
from PIL import Image

DISEASES = {
    "dia": "dia",
    "mal": "mal",
    "oct": "oct"
}


st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("DISEASE PREDICTOR")

image = st.file_uploader("Choose an image")

disease = st.selectbox("Choose the style", [i for i in DISEASES.keys()])

if st.button("Analyze"):
    if image is not None and disease is not None:
        files = {"file": image.getvalue()}
        res = requests.post(f"http://backend:5000/{disease}", files=files)
        img_path = res.json()
        image = Image.open(img_path.get("name"))
        st.image(image)