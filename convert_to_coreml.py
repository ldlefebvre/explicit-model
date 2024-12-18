import coremltools as ct
import tensorflow as tf

# Load the best-performing trained model
model = tf.keras.models.load_model("best_nsfw_safe_model.h5")  # Use BEST_MODEL_NAME

# Convert to Core ML format with dynamic input sizes
mlmodel = ct.convert(model, inputs=[ct.ImageType(shape=(1, None, None, 3))])
mlmodel.save("NSFWSafeClassifier.mlmodel")
print("Model successfully converted to Core ML format!")


#to run it: python convert_to_coreml.py
