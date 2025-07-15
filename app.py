import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ✅ Only ONE call to set_page_config at the TOP
st.set_page_config(page_title="Traffic Sign Classifier", layout="wide")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("traffic_sign_model.h5")

model = load_model()

# Class names mapping
class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
    'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
    'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
    'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End of no passing for vehicles > 3.5 tons'
]

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Home", "About", "Contact"])

# --- Home Page ---
if page == "Home":
    st.markdown("""
        <div style='background: linear-gradient(to right, #74ebd5, #ACB6E5); padding: 30px; border-radius: 10px;'>
            <h1 style='text-align: center;'>🚦 Traffic Sign Recognition Web App</h1>
            <h3 style='text-align: center;'>Built by <span style='color:blue;'>Muhammad Jabran</span></h3>
            <p style='text-align: center;'>Upload an image of a traffic sign and this app will predict its class name using a trained ML model.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a Traffic Sign Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = image.resize((32, 32))
        img_array = np.array(img)

        if img_array.shape == (32, 32, 3):
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            prediction = model.predict(img_array)
            class_index = np.argmax(prediction)
            class_name = class_names[class_index]
            st.success(f"**Predicted Class:** {class_name}")
        else:
            st.error("Image must be RGB with dimensions 32x32")

# --- About Page ---
elif page == "About":
    st.markdown("""
        <div style='background: linear-gradient(to right, #6a11cb, #2575fc); padding: 50px; border-radius: 15px; color: white;'>
            <h1 style='text-align: center;'>About This Project</h1>
            <p style='font-size: 18px;'>This is a deep learning-based traffic sign classification web app trained using a CNN model. 
            It helps recognize 43 different types of traffic signs using image input. The app was developed using Python, TensorFlow, and Streamlit to provide real-time predictions in a browser-based interface.</p>
            <img src='https://cdn.pixabay.com/photo/2017/08/30/07/52/road-sign-2692870_1280.jpg' width='500' style='display:block; margin:auto; border-radius:10px; margin-top:20px;'>
        </div>
    """, unsafe_allow_html=True)

# --- Contact Page ---
elif page == "Contact":
    st.markdown("""
        <div style='background: linear-gradient(to right, #FF416C, #FF4B2B); padding: 50px; border-radius: 15px; color: white;'>
            <h1 style='text-align: center;'>Contact Developer</h1>
            <p style='font-size: 18px; text-align:center;'>Have feedback or suggestions?</p>
            <ul style='font-size: 18px;'>
                <li>Email: jabran.ai@gmail.com</li>
                <li>GitHub: github.com/jabranml</li>
                <li>Location: Pakistan</li>
            </ul>
            <img src='https://cdn.pixabay.com/photo/2016/12/27/21/03/road-sign-1930381_1280.jpg' width='500' style='display:block; margin:auto; border-radius:10px; margin-top:20px;'>
        </div>
    """, unsafe_allow_html=True)

