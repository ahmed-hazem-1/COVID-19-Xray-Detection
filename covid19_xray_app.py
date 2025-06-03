import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="COVID-19 X-ray Detection",
    page_icon="🫁",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = 'COVID-19 Xray Detection.h5'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please make sure the model is in the correct location.")
        return None
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_data):
    """Preprocess the image to match model requirements"""
    # Convert to PIL Image
    img = Image.open(io.BytesIO(image_data))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Check if image is grayscale or RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Convert RGB to grayscale
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Resize to 224x224
    img_resized = cv2.resize(img_gray, (224, 224))
    
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    
    # Add batch and channel dimensions
    img_input = img_normalized.reshape(1, 224, 224, 1)
    
    return img_input, img_resized

def main():
    st.title("COVID-19 X-ray Detection")
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a Convolutional Neural Network (CNN) "
        "to detect COVID-19 from chest X-ray images. Upload an X-ray "
        "image to get a prediction."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        """
        1. Upload a chest X-ray image (PNG, JPG, JPEG)
        2. Wait for the model to process the image
        3. View the prediction result
        
        Note: For best results, use clear frontal chest X-ray images.
        """
    )
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("Model could not be loaded. Please check if the model file exists.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## Upload X-ray Image")
        uploaded_file = st.file_uploader("Choose a chest X-ray image...", 
                                         type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image_data = uploaded_file.read()
            st.image(image_data, caption="Uploaded X-ray Image", use_column_width=True)
            
            # Add a prediction button
            predict_button = st.button("Predict")
            
            if predict_button:
                with st.spinner("Processing..."):
                    # Preprocess the image
                    img_input, img_display = preprocess_image(image_data)
                    
                    # Make prediction
                    prediction = model.predict(img_input)[0][0]
                    
                    # Display results in the second column
                    with col2:
                        st.markdown("## Prediction Result")
                        
                        # Show preprocessed image
                        st.image(img_display, caption="Preprocessed Image (224x224, Grayscale)", 
                                 use_column_width=True)
                        
                        # Display prediction
                        st.markdown("### Diagnosis")
                        prediction_percentage = prediction * 100
                        
                        if prediction > 0.5:
                            st.error(f"COVID-19 Positive (Confidence: {prediction_percentage:.2f}%)")
                            st.markdown("""
                                ⚠️ This result suggests the X-ray shows signs consistent with COVID-19.
                                
                                **Important Note**: This is not a medical diagnosis. Please consult a healthcare professional.
                            """)
                        else:
                            st.success(f"COVID-19 Negative (Confidence: {(100-prediction_percentage):.2f}%)")
                            st.markdown("""
                                ✅ This result suggests the X-ray does not show signs consistent with COVID-19.
                                
                                **Important Note**: This is not a medical diagnosis. Please consult a healthcare professional.
                            """)
                        
                        # Show prediction probability
                        st.markdown("### Prediction Probability")
                        prob_df = {
                            "COVID-19 Positive": prediction_percentage,
                            "COVID-19 Negative": 100 - prediction_percentage
                        }
                        st.progress(prediction)
                        
    # Add disclaimer at the bottom
    st.markdown("---")
    st.markdown("""
    ## Disclaimer
    
    This application is for educational and research purposes only. It is not intended for medical diagnosis. 
    The predictions made by this model should not be used as a substitute for professional medical advice, 
    diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider 
    with any questions you may have regarding a medical condition.
    
    ## About the Model
    
    This model is a Convolutional Neural Network (CNN) trained on chest X-ray images to detect radiographic 
    findings associated with COVID-19. The model was trained on a dataset containing both COVID-19 positive 
    and negative X-ray images.
    
    - **Input**: Grayscale chest X-ray images (224x224 pixels)
    - **Output**: Probability of COVID-19 presence
    """)

if __name__ == "__main__":
    main()
