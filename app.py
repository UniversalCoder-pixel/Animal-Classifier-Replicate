import streamlit as st
from predict import predict_image
from PIL import Image

st.title("🐾 Animal Detector Demo")
st.write("Upload an image to detect the animal!")

uploaded = st.file_uploader("Choose an image", type=['jpg','jpeg','png'])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")
    prediction = predict_image(uploaded)
    st.success(f"Detected animal: **{prediction}**")
