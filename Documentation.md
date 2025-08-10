
# 🐟 Interactive Category Visualization from Hierarchical Dataset

This project involves preprocessing and visualizing a dataset of fish and seafood categories that are structured in a hierarchical text format (e.g., `"fish sea_food trout"`). The goal is to explore and interactively visualize the relationships between high-level and specific fish types using Plotly.

---

## 📁 Dataset Overview

The dataset consists of tuples in a TensorFlow format like:

```python
(b"fish sea_food trout", label)
```

Each item is a category path with one to three levels:
- **Level 1**: Broad category (e.g., `animal`, `fish`)
- **Level 2**: Sub-category (e.g., `sea_food`)
- **Level 3**: Specific type (e.g., `shrimp`, `trout`)

---

## 🧹 Preprocessing Steps

1. **Tensor Conversion**: Since the dataset is in TensorFlow (`EagerTensor`), each string is converted using:

```python
str(text.numpy().decode("utf-8"))
```

2. **Tokenization**: Each string is split into words using `.split()`.

3. **Truncation/Padding**: To ensure consistent formatting, each label is limited to 3 levels. Missing levels are filled with empty strings.

4. **DataFrame Construction**: A pandas `DataFrame` is created with columns: `Level1`, `Level2`, `Level3`.

---

## 📊 Visualization 1: Sunburst Chart

An interactive **sunburst chart** is created using Plotly to show the full category hierarchy:

```python
fig1 = px.sunburst(df, path=["Level1", "Level2", "Level3"], title="Category Hierarchy")
fig1.show()
```

✅ This chart allows zooming and hovering to explore how specific fish types fall under broader categories.

---

## 📊 Visualization 2: Bar Chart of Fish Type Frequency

A bar chart displays how often each specific fish type (Level 3) appears in the dataset:

```python
level3_counts = df[df["Level3"] != ""]["Level3"].value_counts().reset_index()
level3_counts.columns = ["Level3", "count"]

fig2 = px.bar(level3_counts, x="Level3", y="count", title="Fish Types Frequency")
fig2.show()
```

✅ This allows quick insights into the most common types of fish or seafood.

---

## 💾 Saving Charts

Charts can be saved as:

- **Interactive HTML** (best for sharing):
  ```python
  fig.write_html("chart.html")
  ```

- **Static Images** (requires `kaleido`):
  ```python
  fig.write_image("chart.png")
  ```

---

## 📦 Dependencies

Make sure to install:

```bash
pip install pandas plotly kaleido
```

---

## ✅ Summary

- Extracted hierarchical labels from TensorFlow dataset.
- Parsed into Level1, Level2, and Level3.
- Visualized:
  - Full category hierarchy via sunburst.
  - Fish type distribution via bar chart.
- Saved charts for sharing and presentation.

This approach is useful for analyzing structured label data, taxonomy classification, and exploratory data analysis in machine learning datasets.



--------------------------------------------------------------------------------------------------------

# CNN from Scratch - Summary

This notebook implements a **Custom Convolutional Neural Network (CNN)** from scratch using TensorFlow/Keras.

## Workflow

1. **Import Libraries**  
   Essential packages for deep learning, data handling, and visualization are imported.

2. **Declare Constants**  
   Set dataset paths, image dimensions, batch size, and training parameters.

3. **Load Dataset**  
   Training, validation, and test datasets are loaded from directories.

4. **Data Preview**  
   Display sample images from the dataset to verify data loading.

5. **Data Pipeline Optimization**
   - `cache()` → caches dataset in memory after the first epoch.
   - `shuffle(1000)` → shuffles the dataset to prevent learning order.
   - `prefetch(tf.data.AUTOTUNE)` → enables asynchronous data loading.

6. **Image Preprocessing**  
   Images are resized and rescaled for model compatibility.

7. **Build Model**  
   A custom CNN architecture is defined, including convolution, pooling, and dense layers.

8. **Model Summary**  
   Displays the architecture, parameters, and layer details.

9. **Compile Model**  
   Defines optimizer, loss function, and metrics.

10. **Train Model**  
    Model is trained with validation monitoring.

11. **Evaluation**  
    Tested on the test dataset to compute accuracy and loss.

12. **Visualization**
    - Training vs Validation Accuracy & Loss curves
    - Predictions on sample test images

13. **Save Model**  
    Model saved for future inference.

14. **Performance Metrics**
    - ROC-AUC Curve (Multiclass One-vs-Rest)
    - Precision-Recall Curve
    - F1-score vs Threshold

---
**End of Summary**
--------------------------------------------------------------------------------------------------------

# Transfer Learning Experiments - Detailed Summary

## Transfer Learning Models

## Import libraries

- This block imports required libraries such as TensorFlow, Keras application models (MobileNetV2, ResNet50, EfficientNetB0, InceptionV3, VGG16), image preprocessing tools, callbacks, Matplotlib for visualization, and OS utilities.

## Declare Constants

- This block defines constants like image size, batch size, number of channels, number of epochs, and input shape for the neural network.

## Load Train Dataset

- This block defines constants like image size, batch size, number of channels, number of epochs, and input shape for the neural network.

## Load Test Datset

- This block defines constants like image size, batch size, number of channels, number of epochs, and input shape for the neural network.

## Load Validation Dataset

- This block defines constants like image size, batch size, number of channels, number of epochs, and input shape for the neural network.

- This block performs a specific step related to transfer learning model training or evaluation.

- This block performs a specific step related to transfer learning model training or evaluation.

## Prepare Datasets for Performance

- This block performs a specific step related to transfer learning model training or evaluation.

## Load PreTrained Model

### The build_model function:

- Loads a pre-trained CNN model (like MobileNet, ResNet, etc.) without the top classification layer (include_top=False).

- Freezes the base model’s weights (no training).

- Adds custom layers on top:

    - GlobalAveragePooling2D

    - Dropout (to reduce overfitting)

    - Dense softmax output layer for classification.

- Compiles the model with:

    - adam optimizer

    - sparse_categorical_crossentropy loss

    - accuracy metric

### Returns the final compiled model.

- This block defines constants like image size, batch size, number of channels, number of epochs, and input shape for the neural network.

## List of Pretrained Models to Try

- This block performs a specific step related to transfer learning model training or evaluation.

## Train Each Model and Collect Results

- This block defines constants like image size, batch size, number of channels, number of epochs, and input shape for the neural network.

## Get the best Model

- This block performs a specific step related to transfer learning model training or evaluation.

## Rebuild the Best Model (for test + fine-tuning)

- This block defines constants like image size, batch size, number of channels, number of epochs, and input shape for the neural network.

## Re-evaluate Best Model on Test Data

- This block trains the model on the dataset with specified parameters, using callbacks like early stopping and model checkpointing.

## Fine-Tune the Best Model

- This block compiles the model with optimizer, loss function, and evaluation metrics.

## Compare Before & After Fine-Tuning

- This block performs a specific step related to transfer learning model training or evaluation.

## Extract metrics from fine-tuning history

- This block performs a specific step related to transfer learning model training or evaluation.

## Accuracy and Validation Plot

- This block visualizes training and validation accuracy/loss over epochs using Matplotlib.

## Making prediction on single image from test dataset

- This block imports required libraries such as TensorFlow, Keras application models (MobileNetV2, ResNet50, EfficientNetB0, InceptionV3, VGG16), image preprocessing tools, callbacks, Matplotlib for visualization, and OS utilities.

## Make Predictions

- This block performs a specific step related to transfer learning model training or evaluation.

## Display 9 images with predictions

- This block visualizes training and validation accuracy/loss over epochs using Matplotlib.

# Performance Metrices

- This block imports required libraries such as TensorFlow, Keras application models (MobileNetV2, ResNet50, EfficientNetB0, InceptionV3, VGG16), image preprocessing tools, callbacks, Matplotlib for visualization, and OS utilities.

- This block evaluates the trained model on the validation or test dataset.

## Collecting prediction probabilities from best_model on the entire test dataset

- This block performs a specific step related to transfer learning model training or evaluation.

## ROC-AUC Curve (Multiclass OvR)

- The One-vs-Rest (OvR) strategy is often used with ROC-AUC curves to evaluate model performance. Each class is treated as a separate binary problem against all other classes. This results in an ROC curve and AUC score for each class, allowing for a more nuanced understanding of how well the model distinguishes each class. 

## Precission-Recall Curve

- A precision-recall curve (PR curve) is a graphical tool used to evaluate the performance of binary classification models, especially when dealing with imbalanced datasets. It visualizes the trade-off between precision and recall at various classification thresholds. The curve is plotted with recall on the x-axis and precision on the y-axis. 

## F1-score vs. Threshold

- The F1-score and threshold are both important concepts in machine learning, particularly in binary classification problems. The F1-score is a metric that balances precision and recall, providing a single measure of a model's accuracy, while the threshold determines how the model's output (typically a probability) is converted into a class prediction. 

## Save Final Best Model

- This block defines the transfer learning model architecture using a pretrained network as the base model and adding custom classification layers.

--------------------------------------------------------------------------------------------------------

## 1. Transfer Learning (Fine-Tuned) Model
- Accuracy: 100%

- Precision / Recall / F1-score: All are 1.00 for almost every class (except minor recall drop for “animal fish bass” with recall = 0.92).

- Confusion Matrix: Nearly perfect classification — very few misclassifications.

## 2. Custom CNN Model
- Accuracy: 97%

- Precision / Recall / F1-score: Slightly lower scores; one class (“animal fish bass”) has 0% precision/recall (model failed to classify it correctly).

- Confusion Matrix: Shows multiple misclassifications across several classes.

## Recommendation
### Choose the Transfer Learning (Fine-Tuned) Model because:

- It achieves higher overall accuracy (100% vs 97%).

- It maintains high performance across all classes, including minority classes.

- It shows minimal confusion between categories, which is crucial for multiclass classification.

## Transfer learning leverages pre-trained weights, which likely helped in achieving better generalization, especially for smaller datasets or rare classes.


--------------------------------------------------------------------------------------------------------
# Business Perspective – CNN Multiclass Fish Image Classification
## 1. Problem Statement & Value Proposition
- In industries like fisheries, aquaculture, food processing, and environmental monitoring, accurately identifying fish species is crucial. Traditionally, this requires manual labor, expert knowledge, and is prone to human error.
- A CNN multiclass classification model can automate this process, providing faster, scalable, and consistent fish identification, which saves time, reduces cost, and improves operational efficiency.

## 2. Key Business Use Cases
- Fisheries & Aquaculture – Automatic sorting of fish by species for packaging and distribution.

- Quality Control in Food Processing – Ensuring correct labeling and reducing species misidentification (which can cause legal and regulatory issues).

- Wildlife & Marine Conservation – Tracking species diversity and population using underwater cameras.

- Fishing Regulation Compliance – Detecting protected or endangered species in real-time to prevent overfishing.

- Supply Chain Transparency – Ensuring customers get the exact fish species promised (avoiding fraud).

## 3. Business Benefits
### Benefit	                            Impact
#### Efficiency	        Eliminates manual sorting, allowing 24/7 operation.
#### Accuracy	        Reduces species misidentification errors to near zero.
#### Cost Savings	    Cuts labor costs and prevents fines from regulatory violations.
#### Scalability	    Can process thousands of images in minutes for large-scale operations.
#### Brand Trust	    ccurate species labeling increases customer trust and loyalty.

## 4. Revenue Opportunities
- Selling AI-based sorting systems to fish farms and seafood processing plants.

- Licensing the classification API to seafood distributors and research institutions.

- Providing real-time monitoring solutions for governments and NGOs involved in marine conservation.

- Integration into fishing vessel equipment for onboard species detection.

## 5. Risks & Challenges
- Dataset Bias – Poor training data can lead to misclassification in rare species.

- Lighting/Environmental Variations – Underwater lighting and fish movement can reduce accuracy.

- Maintenance Cost – Model retraining is needed as new species are introduced or visual patterns change.

- Hardware Dependency – High-resolution image capture and GPU-based inference may be costly.

## 6. Business Impact Example
- Imagine a large seafood exporter that processes 10,000 fish/day:

- Manual sorting = 5 workers × $300/month × 12 months = $18,000/year.

- AI model reduces workforce by 80%, saving ~$14,400/year.

- Additionally, avoiding mislabeling fines worth $5,000/year and increasing operational speed by 3×.

- Total first-year ROI = $14,400 + $5,000 = $19,400 savings, excluding the value of faster throughput.