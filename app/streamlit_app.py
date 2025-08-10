import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the best fine-tuned model
@st.cache_resource
def load_model():
    # Try multiple possible paths for the model
    possible_paths = [
        "models/Transfer_Learning_Model.keras",  # Same directory
        #"../models/Transfer_Learning_Model.keras",  # models subdirectory
        # "../models/Custom_CNN_Fish_Classificatio.keras",  # models subdirectory
        # "models/Custom_CNN_Fish_Classificatio.keras",  # models subdirectory
    ]
    
    model = None
    model_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                model_path = path
                st.success(f"✅ Model loaded successfully from: {path}")
                break
            except Exception as e:
                st.warning(f"⚠️ Failed to load model from {path}: {str(e)}")
                continue
    
    if model is None:
        st.error("❌ Could not load model from any of the expected paths.")
        st.info("Please ensure one of these files exists:")
        for path in possible_paths:
            st.write(f"- {path}")
        st.stop()
    
    return model, model_path

# Try to load model
try:
    model, loaded_path = load_model()
except Exception as e:
    st.error(f"Fatal error loading model: {str(e)}")
    st.stop()

# Define class names (based on your training order)
class_names = [
    'Animal Fish',
    'Animal Fish Bass',
    'Fish Sea_food Black_sea_sprat',
    'Fish Sea_food gilt_head_bream',
    'Fish Sea_food hourse_mackerel',
    'Fish Sea_food red_mullet',
    'Fish Sea_food red_sea_bream',
    'Fish Sea_food sea_bass',
    'Fish Sea_food shrimp',
    'Fish Sea_food striped_red_mullet',
    'Fish Sea_food trout'
]

# Set image size
IMAGE_SIZE = (224, 224)

st.title("🐟 Multiclass Fish Image Classification")
st.markdown("Upload a fish image to predict the fish class using a fine-tuned transfer learning model.")

# Show model info
with st.expander("ℹ️ Model Information"):
    st.write(f"**Loaded model:** {loaded_path}")
    st.write(f"**Number of classes:** {len(class_names)}")
    st.write(f"**Input image size:** {IMAGE_SIZE}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        with st.spinner("🔍 Analyzing image..."):
            # Preprocess the image
            image_resized = image.resize(IMAGE_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(image_resized)
            img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
            
            # Normalize pixel values to [0,1] - important for many models
            img_array = img_array / 255.0

            # Predict
            try:
                predictions = model.predict(img_array, verbose=0)
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = round(100 * np.max(predictions[0]), 2)
                
                # Display results
                st.markdown(f"### 🎯 Prediction: `{predicted_class}`")
                st.markdown(f"### 📊 Confidence: `{confidence}%`")
                
                # Color-code confidence
                if confidence > 90:
                    st.success("High confidence prediction!")
                elif confidence > 70:
                    st.info("Medium confidence prediction")
                else:
                    st.warning("Low confidence prediction - consider using a clearer image")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

    # Show probability distribution
    if 'predictions' in locals():
        st.subheader("📈 Class Probabilities")
        
        # Create a more readable version of class names for display
        display_names = [name.replace('Fish Sea_food ', '').replace('Animal Fish', 'Fish').replace('_', ' ').title() 
                        for name in class_names]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(class_names)), predictions[0], color='skyblue', alpha=0.7)
        
        # Highlight the predicted class
        max_idx = np.argmax(predictions[0])
        bars[max_idx].set_color('orange')
        
        ax.set_ylabel("Probability")
        ax.set_xlabel("Fish Classes")
        ax.set_ylim([0, 1])
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(display_names, rotation=45, ha='right')
        
        # Add probability values on bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                   f'{yval:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)

else:
    # Show sample information when no image is uploaded
    st.info("👆 Please upload a fish image to get started!")
    
    st.subheader("📚 Supported Fish Classes:")
    for i, class_name in enumerate(class_names):
        display_name = class_name.replace('Fish Sea_food ', '').replace('Animal Fish', 'Fish').replace('_', ' ').title()
        st.write(f"{i+1}. {display_name}")