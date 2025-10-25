import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Gastrointestinal Disease Detector",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c7bb6;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f7ff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .disease-name {
        font-size: 1.8rem;
        font-weight: bold;
        color: #d9534f;
    }
    .confidence {
        font-size: 1.2rem;
        color: #5cb85c;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .placeholder-container {
        width: 100%;
        height: 300px;
        display: flex;
        justify-content: center;
        align-items: center;
        border: 2px dashed #ccc;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .placeholder-text {
        color: #888;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🏥 Gastrointestinal Disease Detection</h1>', unsafe_allow_html=True)

# Information about gastrointestinal diseases
with st.expander("ℹ️ About Gastrointestinal Diseases and This Tool"):
    st.markdown("""
    ### Gastrointestinal Tract Diseases
    The gastrointestinal (GI) tract plays a crucial role in digestion and nutrient absorption. 
    Diseases affecting the GI tract can significantly impact quality of life and require accurate diagnosis for proper treatment.
    
    This tool uses a deep learning model to classify endoscopic images into one of 8 categories:
    
    1. **dyed-lifted-polyps** - Polyps that have been dyed and lifted for removal
    2. **dyed-resection-margins** - Margins of resected tissue that have been dyed
    3. **esophagitis** - Inflammation of the esophagus
    4. **normal-cecum** - Healthy cecum appearance
    5. **normal-pylorus** - Healthy pylorus appearance
    6. **normal-z-line** - Healthy z-line appearance
    7. **ulcerative-colitis** - Chronic inflammatory bowel disease
    8. **polyps** - Abnormal tissue growths
    
    ### How to Use This Tool
    1. Upload an endoscopic image of the gastrointestinal tract
    2. The AI model will analyze the image
    3. Review the prediction results and confidence level
    4. Always consult a healthcare professional for medical diagnosis
    """)

# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('resNet.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_trained_model()

# Updated class labels based on your model's training classes
class_labels = [
    "dyed-lifted-polyps", 
    "dyed-resection-margins", 
    "esophagitis", 
    "normal-cecum", 
    "normal-pylorus", 
    "normal-z-line", 
    "polyps",
    "ulcerative-colitis" 
]

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Adjust based on your model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if your model was trained with normalized images
    return img_array

# Function to make prediction
def predict_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions)
    return predicted_class, confidence, predictions

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload an endoscopic image using the file uploader
    2. Supported formats: JPG, JPEG, PNG
    3. Click the "Analyze Image" button to process
    4. Results will show the predicted condition and confidence level
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File uploader
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an endoscopic image...", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

with col2:
    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_container_width=True)
        
        # Make prediction when button is clicked
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            if model is not None:
                with st.spinner("Analyzing image..."):
                    # Preprocess the image
                    processed_image = preprocess_image(image_display)
                    
                    # Make prediction
                    predicted_class, confidence, all_predictions = predict_image(model, processed_image)
                    
                    # Display results - ONLY THE HIGHEST PREDICTION
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown("### 📊 Prediction Results")
                    st.markdown(f'<p class="disease-name">Predicted: {class_labels[predicted_class[0]]}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="confidence">Confidence: {confidence*100:.2f}%</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            else:
                st.error("Model could not be loaded. Please check if 'mass.h5' is in the correct directory.")
    else:
        # Improved placeholder with custom CSS
        st.markdown(
            """
            <div class="placeholder-container">
                <p class="placeholder-text">Image preview will appear here</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

# Disclaimer
st.markdown("---")
st.markdown("""
<div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107;">
<strong>⚠️ Important Disclaimer:</strong> This tool is for educational and research purposes only. 
It is not a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of qualified healthcare providers with questions about medical conditions.
</div>
""", unsafe_allow_html=True)