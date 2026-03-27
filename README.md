# Furina Fanart Sorter (AI Image Classifier)
## URL: https://furina-fanart-image-classifier-r2cgvmhbbhz6ysemn9gtvy.streamlit.app/

## Overview
An AI-powered web application built with Python and Streamlit that automatically sorts and classifies Genshin Impact fanart. 

This project uses a custom-trained **Multi-Class Image Classification** model (MobileNet via TensorFlow/Keras) to distinguish the character "Furina" from other visually similar Hydro characters, while also successfully identifying and rejecting completely unrelated images.

## Features
* **Drag & Drop Interface:** Easily upload images directly from your computer.
* **Instant AI Analysis:** Processes the image and outputs a confidence score in real-time.

## How the AI Works
The model was intentionally trained to avoid the "Forced Choice" binary classification problem by utilizing three distinct categories:
1. **Furina:** The target character.
2. **Not Furina (Hard Negatives):** Exclusively other blue/water-themed characters (e.g., Neuvillette, Yelan) to force the AI to learn specific character features rather than just color palettes.
3. **Junk Drawer (OOD):** A diverse set of real-world photography and unrelated art to teach the model how to reject out-of-distribution (OOD) images safely.

## Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
* **Model Training:** [Teachable Machine](https://teachablemachine.withgoogle.com/)
* **Image Processing:** [Pillow (PIL)](https://python-pillow.org/) & [NumPy](https://numpy.org/)
