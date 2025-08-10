import tensorflow as tf
import numpy as np
import io
from PIL import Image

CLASS_NAMES = [
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

async def load_model_and_predict(file, model_type):
     # Load model
    model_path = f"../models/Transfer_Learning_Model.keras"
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))  # adjust to your model's input
    img_array = tf.keras.utils.img_to_array(image) / 224.0
    img_array = tf.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    top_class = np.argmax(predictions)
    confidence = float(predictions[top_class])

    return {
        "predicted_class": CLASS_NAMES[top_class],
        "confidence": round(confidence, 3),
        "class_probabilities": dict(zip(CLASS_NAMES, map(float, predictions)))
    }