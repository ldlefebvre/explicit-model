import coremltools as ct
from PIL import Image
import numpy as np

# Path to your Core ML model (mlpackage)
model_path = "NSFWSafeClassifier.mlpackage"

# Load the Core ML model
print("Loading Core ML model...")
mlmodel = ct.models.MLModel(model_path)
print(f"Model loaded successfully: {mlmodel}")

# Path to the test image
image_path = "1.jpg"

# Open the image
print(f"Opening image: {image_path}...")
image = Image.open(image_path).convert("RGB")

# Resize image to the required size (512x512)
print("Preprocessing image...")
image = image.resize((512, 512))  # Resize to 512x512

# Prepare input dictionary for Core ML using the correct input name
input_dict = {'inputs': image}  # 'inputs' is the expected key

# Run the model
print("Running the model...")
predictions = mlmodel.predict(input_dict)

# Extract predictions
print("Predictions:")
class_label = predictions["classLabel"]  # Predicted class label
class_probs = predictions["classLabel_probs"]  # Dictionary of probabilities

print(f"Predicted Class: {class_label}")
print("Class Probabilities:")
for label, confidence in class_probs.items():
    print(f"Label: {label}, Confidence: {confidence}")
