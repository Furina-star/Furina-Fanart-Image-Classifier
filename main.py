import os

# Set the environment variable to use the legacy Keras API
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set up the Streamlit app
st.set_page_config(page_title="Furina Fanart Sorter", page_icon="🌊")

st.title("🌊 Furina Fanart Sorter")
st.write("Upload an image to see if its Furina or not!")

# Get the Model
@st.cache_resource
def load_ai_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names


model, class_names = load_ai_model()

st.divider()
st.subheader("📥 Submit Your Artwork")
st.markdown("""
* **Click** the box to browse your files.
* **Drag & Drop** an image from your computer.
""")

# The single native uploader that handles everything
uploaded_file = st.file_uploader("Drop or Paste Image Here", type=["jpg", "jpeg", "png", "webp"])

# User Interface
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Analyzed Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        # Teachable Machine Image Processing
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make the prediction
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()[2:]
        confidence_score = prediction[0][index]

    # Show the results
    st.divider()

    # We now have 3 distinct categories to handle!
    if class_name == "Furina":
        st.success(f"**Result: Furina!** 🌊")
        st.write(f"Confidence: {confidence_score * 100:.1f}%")
        st.balloons()

    elif class_name == "Not Furina":
        st.info(f"**Result: Not Furina** 💧")
        st.write(f"Confidence: {confidence_score * 100:.1f}%")

    else:
        # This catches the Junk Drawer!
        st.info(f"**Result: Not Furina** 💧")
        st.write(f"Confidence: {confidence_score * 100:.1f}%")
