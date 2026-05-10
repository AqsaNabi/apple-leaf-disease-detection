import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

st.write("App started successfully")
# load model
#model = tf.keras.models.load_model("Modelplant_newplant_aspp_plantdetection.h5")

# class labels
classes = ["Healthy", "Apple Scab", "Black Rot", "Leaf Blotch"]

st.title("🍎 Apple Leaf Disease Detection")

# upload image
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
