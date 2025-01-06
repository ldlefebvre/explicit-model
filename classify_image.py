import tensorflow as tf
import numpy as np
from PIL import Image

# Path to the TensorFlow Keras model
keras_model_path = "best_nsfw_safe_model.keras"

# Load the Keras model
print("Loading Keras model...")
model = tf.keras.models.load_model(keras_model_path)
print(f"Model loaded successfully: {model}")

# Path to the test image
image_path = "1.jpg"

# Open the image
print(f"Opening image: {image_path}...")
image = Image.open(image_path).convert("RGB")

# Resize image to match the model input (512x512)
print("Preprocessing image...")
image = image.resize((512, 512))
image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values

# Add batch dimension to the image
input_image = np.expand_dims(image_array, axis=0)

# Predict using the Keras model
print("Running prediction on the Keras model...")
predictions = model.predict(input_image)

# Extract predictions
print("Predictions:")
print(f"NSFW Confidence: {predictions[0][0]}")
print(f"Safe Confidence: {predictions[0][1]}")

# Determine the final classification
final_label = "NSFW" if predictions[0][0] > predictions[0][1] else "Safe"
print(f"Final Label: {final_label}")
