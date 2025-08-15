import streamlit as st
from backend import load_model, predict_image, load_sample_images
import matplotlib.pyplot as plt
import cv2
import os

# ======= PAGE CONFIG =======
st.set_page_config(page_title="Skin Disease Classifier", layout="centered")

# ======= CUSTOM CSS (Brown & Gold) =======
st.markdown("""
<style>
body {
    background-color: #f9f5f0;
    color: #3e2723;
}
h1, h2, h3 {
    color: #6d4c41;
}
.stButton>button {
    background-color: #a1887f;
    color: white;
    border-radius: 10px;
}
.stButton>button:hover {
    background-color: #8d6e63;
}
</style>
""", unsafe_allow_html=True)

# ======= TITLE =======
st.title("🩺 Skin Disease Classifier")
st.markdown("Upload an image of a skin lesion to get an instant prediction. **Model: ResNet50 (ImageNet)**")

# ======= LOAD MODEL =======
model = load_model()

# ======= FILE UPLOAD =======
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")
    preds = predict_image(model, uploaded_file)
    st.subheader("Prediction Results")
    st.bar_chart(preds)

# ======= SAMPLE IMAGES =======
if st.button("Try with sample images"):
    samples = load_sample_images()
    for img_path in samples:
        st.image(img_path, caption=os.path.basename(img_path))
        preds = predict_image(model, img_path)
        st.bar_chart(preds)
