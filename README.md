# 🐟 AI-Powered Fish Species Classification System

A comprehensive deep learning project for automated fish and seafood species identification using Convolutional Neural Networks (CNNs) and Transfer Learning techniques.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Structure](#dataset-structure)
- [Features](#features)
- [Technical Approach](#technical-approach)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)


## 🎯 Project Overview

This project implements an end-to-end machine learning solution for fish species classification using computer vision. The system can automatically identify and categorize different types of fish and seafood from images, providing valuable automation for fisheries, aquaculture, and food processing industries.

### Key Objectives
- Develop accurate fish species classification models
- Compare custom CNN vs Transfer Learning approaches
- Create interactive visualizations for data exploration
- Provide business-ready deployment solutions
- Achieve production-level accuracy and reliability

## 📊 Dataset Structure

The dataset consists of hierarchical labels in TensorFlow format:

```python
(b"fish sea_food trout", label)
```

### Hierarchy Levels
- **Level 1**: Broad category (e.g., `animal`, `fish`)
- **Level 2**: Sub-category (e.g., `sea_food`)
- **Level 3**: Specific type (e.g., `shrimp`, `trout`, `bass`)

### Data Preprocessing Pipeline
1. **Tensor Conversion**: Convert TensorFlow EagerTensor to strings
2. **Tokenization**: Split category paths into individual components
3. **Standardization**: Ensure consistent 3-level hierarchy
4. **DataFrame Construction**: Organize data for analysis and modeling

## ✨ Features

- **Interactive Web Application**: Streamlit-based interface for easy model interaction
- **Jupyter Notebook Workflow**: Step-by-step analysis and model development
- **Custom CNN Architecture**: Built-from-scratch convolutional neural network
- **Transfer Learning Models**: Leveraging pre-trained networks (MobileNetV2, ResNet50, EfficientNetB0, InceptionV3, VGG16)
- **Model Comparison Tools**: Comprehensive evaluation and performance analysis
- **Interactive Visualizations**: Sunburst charts and frequency analysis
- **Performance Metrics**: ROC-AUC curves, Precision-Recall analysis, F1-score optimization
- **Prediction Utilities**: Easy-to-use inference functions
- **Fine-tuning Capabilities**: Advanced transfer learning with custom layers

## 🔬 Technical Approach

### 1. Custom CNN Implementation
```python
# Key components:
- Convolutional layers with ReLU activation
- Max pooling for feature extraction
- Dropout layers for regularization
- Dense layers for classification
- Softmax output for multi-class prediction
```

### 2. Transfer Learning Pipeline
```python
def build_model(base_model, num_classes):
    # Load pre-trained model without top layer
    base = base_model(weights='imagenet', include_top=False)
    base.trainable = False  # Freeze weights
    
    # Add custom classification head
    model = tf.keras.Sequential([
        base,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    return model
```

### 3. Data Optimization Techniques
- **Caching**: Store preprocessed data in memory
- **Prefetching**: Asynchronous data loading
- **Shuffling**: Prevent order-based learning bias
- **Augmentation**: Increase dataset diversity

## 🏆 Model Performance

### Transfer Learning (Fine-Tuned) Model - **RECOMMENDED**
- **Accuracy**: 100%
- **Precision/Recall/F1-score**: 1.00 for nearly all classes
- **Key Advantage**: Superior performance on minority classes
- **Business Impact**: Near-perfect classification with minimal errors

### Custom CNN Model
- **Accuracy**: 97%
- **Performance**: Good overall, some challenges with rare species
- **Use Case**: Educational purposes and understanding CNN fundamentals

### Performance Comparison

| Metric | Transfer Learning | Custom CNN |
|--------|------------------|------------|
| Overall Accuracy | 100% | 97% |
| Training Speed | Faster | Slower |
| Resource Usage | Moderate | High |
| Generalization | Excellent | Good |

## 🛠 Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- CUDA-compatible GPU (recommended)

### Required Dependencies
```bash
pip install tensorflow>=2.8.0
pip install pandas
pip install plotly
pip install matplotlib
pip install scikit-learn
pip install streamlit
pip install pillow
pip install numpy
pip install kaleido  # For saving static plots
```

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/fish-classification.git
cd fish-classification

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

## 🚀 Quick Start

### Option 1: Interactive Web App (Recommended for Beginners)
```bash
# Launch the Streamlit application
streamlit run app/streamlit_app.py

# Open your browser to http://localhost:8501
# Upload fish images and get instant predictions!
```

### Option 2: Jupyter Notebooks (For Learning & Development)
```bash
# Start Jupyter Lab
jupyter lab

# Navigate to notebooks/ folder and run:
# 1. 01_data_exploration.ipynb
# 2. 02_custom_cnn.ipynb
# 3. 03_transfer_learning.ipynb
# 4. 04_Model_comparision.ipynb
```

### Option 3: Direct Prediction (For Developers)
```python
from utils.predict import predict_fish_species

# Classify a fish image
result = predict_fish_species('path/to/your/fish.jpg')
print(f"Predicted Species: {result['species']}")
```

### 1. Explore the Notebooks
Start by examining the Jupyter notebooks in sequential order:

```python
# 1. Data exploration and visualization
jupyter notebook notebooks/01_data_exploration.ipynb

# 2. Custom CNN implementation
jupyter notebook notebooks/02_custom_cnn.ipynb

# 3. Transfer learning experiments
jupyter notebook notebooks/03_transfer_learning.ipynb

# 4. Model comparison and analysis
jupyter notebook notebooks/04_Model_comparision.ipynb
```

### 2. Run the Interactive Web Application
```bash
# Launch Streamlit app for interactive predictions
streamlit run app/streamlit_app.py

# Features:
# - Upload fish images for classification
# - Real-time predictions with confidence scores
# - Model comparison interface
# - Interactive visualizations
```

### 3. Use Prediction Utilities
```python
# Import prediction utilities
from utils.predict import predict_fish_species

# Load and use trained models
prediction = predict_fish_species(
    image_path='path/to/fish_image.jpg',
    model_path='models/transfer_learning_best.keras'
)

print(f"Species: {prediction['species']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### 4. Access Pre-trained Models
```python
import tensorflow as tf

# Load custom CNN model
custom_model = tf.keras.models.load_model('models/custom_cnn.keras')

# Load transfer learning model (recommended)
best_model = tf.keras.models.load_model('models/transfer_learning_best.keras')
```

### 5. View Results and Reports
- Check `results/plots/` for generated visualizations
- Review `results/reports/` for performance metrics
- Read `Documentation.md` for detailed technical information

## 📁 Project Structure

```
fish-classification/
├── data/
│   ├── train/              # Training dataset
│   ├── validation/         # Validation dataset
│   └── test/              # Test dataset
├── models/
│   ├── custom_cnn.keras          # Trained custom CNN model
│   └── transfer_learning_best.keras  # Best transfer learning model
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Dataset analysis and visualization
│   ├── 02_custom_cnn.ipynb          # Custom CNN implementation
│   ├── 03_transfer_learning.ipynb   # Transfer learning experiments
│   └── 04_Model_comparision.ipynb   # Performance comparison analysis
├── utils/
│   └── predict.py         # Utility functions for model inference
├── app/
│   └── streamlit_app.py   # Interactive web application
├── results/
│   ├── plots/             # Generated visualizations and charts
│   └── reports/           # Performance reports and metrics
├── requirements.txt       # Project dependencies
├── Documentation.md       # Detailed technical documentation
└── README.md             # Project overview and setup guide
```

## 📊 Results & Visualizations

### Interactive Data Exploration
- **Sunburst Charts**: Hierarchical species relationships
- **Frequency Analysis**: Distribution of fish types in dataset
- **Geographic Mapping**: Species distribution patterns

### Model Performance Visualizations
- **Training Curves**: Accuracy and loss progression
- **Confusion Matrices**: Classification error analysis
- **ROC-AUC Curves**: Multi-class performance evaluation
- **Precision-Recall Curves**: Detailed performance metrics

### Sample Predictions
The system provides visual prediction outputs showing:
- Original fish image
- Predicted species with confidence scores
- Top-3 alternative predictions
- Classification certainty visualization

