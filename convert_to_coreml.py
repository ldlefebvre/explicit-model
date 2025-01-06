# import coremltools as ct
# import tensorflow as tf
# import gc
# from numba import cuda

# # Load the trained Keras model
# print("Loading the trained Keras model...")
# model = tf.keras.models.load_model("best_nsfw_safe_model.keras")

# # Save the model in TensorFlow SavedModel format
# saved_model_dir = "saved_model"
# print("Saving the model in TensorFlow SavedModel format...")
# tf.saved_model.save(
#     model,
#     saved_model_dir,
#     signatures=tf.function(model.call).get_concrete_function(
#         tf.TensorSpec([None, 512, 512, 3], tf.float32)
#     ),
# )
# print("Model saved in TensorFlow SavedModel format.")

# # Convert the SavedModel to Core ML format and ensure the old .mlmodel format
# print("Converting SavedModel to Core ML format...")
# mlmodel = ct.convert(
#     saved_model_dir,
#     source="tensorflow",
#     inputs=[ct.ImageType(shape=(1, 512, 512, 3))],
#     convert_to="neuralnetwork",  # Ensures .mlmodel format
# )

# # Save the Core ML model in .mlmodel format
# # mlmodel.save("NSFWSafeClassifier.mlmodel")
# mlmodel.save("NSFWSafeClassifier.mlpackage")
# print("Model successfully converted to Core ML format and saved as 'NSFWSafeClassifier.mlmodel'!")

# # ========================
# # Cleanup resources
# # ========================
# print("Cleaning up resources...")
# # Clear TensorFlow session to release memory
# tf.keras.backend.clear_session()

# # Explicitly delete large objects
# del model
# del mlmodel

# # Force garbage collection
# gc.collect()

# # Try to reset GPU memory if numba is installed
# try:
#     from numba import cuda
#     cuda.select_device(0)  # Adjust if using multiple GPUs
#     cuda.close()
#     print("GPU memory cleared.")
# except ModuleNotFoundError:
#     print("Numba not found. Skipping GPU memory cleanup.")

# print("Resources cleaned up. Script finished.")



# import coremltools as ct
# import tensorflow as tf
# import numpy as np
# import gc
# import os

# # ========================
# # Helper Functions
# # ========================

# def log_tensorflow_model(model, input_shape=(1, 512, 512, 3)):
#     """Test TensorFlow model predictions and log outputs."""
#     print("\n==== TensorFlow Model Testing ====")
#     try:
#         # Create a dummy input
#         dummy_input = np.random.rand(*input_shape).astype(np.float32)
#         print(f"Testing model with dummy input of shape {dummy_input.shape}...")
#         output = model.predict(dummy_input)
#         print(f"Model output shape: {output.shape}")
#         print(f"Model output values: {output}")

#         if output.shape[-1] == 2:
#             print(f"Detected 2-class output: Assuming [NSFW, SAFE]")
#         else:
#             print(f"Unexpected output shape: {output.shape}")
#     except Exception as e:
#         print(f"Error testing TensorFlow model: {e}")


# def test_coreml_model(mlmodel, input_shape=(1, 512, 512, 3)):
#     """Test Core ML model predictions and log outputs."""
#     print("\n==== Core ML Model Testing ====")
#     try:
#         # Core ML predict requires macOS 10.13 or later
#         if os.name != "posix":
#             print("Core ML model prediction is only supported on macOS.")
#             return
#         # Create a dummy input
#         # dummy_input = np.random.rand(*input_shape).astype(np.float32)
#         dummy_input = np.random.rand(*input_shape).astype(np.float32) / 255.0
#         input_image = Image.fromarray((dummy_input * 255).astype(np.uint8))
#         print(f"Testing Core ML model with dummy input of shape {dummy_input.shape}...")
#         # predictions = mlmodel.predict({"input": dummy_input})
#         predictions = mlmodel.predict({"inputs": input_image})
#         print(f"Core ML model predictions: {predictions}")
#     except Exception as e:
#         print(f"Error testing Core ML model: {e}")


# def clear_resources(*resources):
#     """Clear memory resources."""
#     print("\n==== Cleaning Up Resources ====")
#     for resource in resources:
#         try:
#             del resource
#         except NameError:
#             print("Resource already cleared.")
#     tf.keras.backend.clear_session()
#     gc.collect()

#     try:
#         from numba import cuda
#         cuda.select_device(0)  # Adjust if using multiple GPUs
#         cuda.close()
#         print("GPU memory cleared.")
#     except ModuleNotFoundError:
#         print("Numba not found. Skipping GPU memory cleanup.")

#     print("Resources cleaned up.")


# # ========================
# # Main Script
# # ========================
# model = None
# mlmodel = None

# try:
#     # Load the trained Keras model
#     print("\n==== Loading TensorFlow Keras Model ====")
#     keras_model_path = "best_nsfw_safe_model.keras"
#     if not os.path.exists(keras_model_path):
#         raise FileNotFoundError(f"Model file not found: {keras_model_path}")
#     model = tf.keras.models.load_model(keras_model_path)
#     print(f"Successfully loaded model from {keras_model_path}")

#     # Test TensorFlow model predictions
#     log_tensorflow_model(model)

#     # Save the model in TensorFlow SavedModel format
#     saved_model_dir = "saved_model"
#     # print("\n==== Saving Model in TensorFlow SavedModel Format ====")
#     # tf.saved_model.save(
#     #     model,
#     #     saved_model_dir,
#     #     signatures=tf.function(model.call).get_concrete_function(
#     #         tf.TensorSpec([None, 512, 512, 3], tf.float32)
#     #     ),
#     # )
#     # print(f"Model successfully saved in TensorFlow SavedModel format: {saved_model_dir}")

#     # Convert the SavedModel to Core ML format
#     print("\n==== Converting to Core ML Format ====")
#     class_labels = ["nsfw", "safe"]
#     mlmodel = ct.convert(
#         saved_model_dir,
#         source="tensorflow",
#         inputs=[ct.ImageType(shape=(1, 512, 512, 3))],
#         classifier_config=ct.ClassifierConfig(class_labels),
#         convert_to="neuralnetwork",  # Ensures .mlmodel format
#     )
#     print("Successfully converted model to Core ML format.")

#     print("\n==== Converting to Core ML Format ====")
#     class_labels = ["nsfw", "safe"]
#     mlmodel = ct.convert(
#         saved_model_dir,
#         source="tensorflow",
#         inputs=[ct.ImageType(shape=(512, 512, 3), color_layout="RGB")],
#         classifier_config=ct.ClassifierConfig(class_labels),
#     )
#     print("Successfully converted model to Core ML format.")

#     # Save the Core ML model in .mlmodel format
#     # coreml_model_path = "NSFWSafeClassifier.mlmodel"
#     # mlmodel.save(coreml_model_path)
#     # print(f"Core ML model saved as: {coreml_model_path}")

#     mlmodel.save("NSFWSafeClassifier.mlpackage")
#     print("Model successfully converted to Core ML format and saved as 'NSFWSafeClassifier.mlpackage'!")

#     # Test Core ML model predictions
#     test_coreml_model(mlmodel)

# except FileNotFoundError as fnfe:
#     print(f"\nERROR: {fnfe}")

# except Exception as e:
#     print(f"\nERROR: {e}")

# finally:
#     # Cleanup resources
#     clear_resources(model, mlmodel)

# print("\n==== Script Finished ====")



# import coremltools as ct
# import tensorflow as tf
# import numpy as np
# import gc
# import os
# from PIL import Image


# # ========================
# # Helper Functions
# # ========================

# def log_tensorflow_model(model, input_shape=(1, 512, 512, 3)):
#     """Test TensorFlow model predictions and log outputs."""
#     print("\n==== TensorFlow Model Testing ====")
#     try:
#         # Create a dummy input
#         dummy_input = np.random.rand(*input_shape).astype(np.float32)
#         print(f"Testing model with dummy input of shape {dummy_input.shape}...")
#         output = model.predict(dummy_input)
#         print(f"Model output shape: {output.shape}")
#         print(f"Model output values: {output}")

#         if output.shape[-1] == 2:
#             print(f"Detected 2-class output: Assuming [NSFW, SAFE]")
#         else:
#             print(f"Unexpected output shape: {output.shape}")
#     except Exception as e:
#         print(f"Error testing TensorFlow model: {e}")


# def test_coreml_model(mlmodel, input_shape=(512, 512, 3)):
#     """Test Core ML model predictions and log outputs."""
#     print("\n==== Core ML Model Testing ====")
#     try:
#         # Core ML predict requires macOS 10.13 or later
#         if os.name != "posix":
#             print("Core ML model prediction is only supported on macOS.")
#             return

#         # Create a dummy input image with pixel values normalized
#         dummy_input = np.random.rand(*input_shape).astype(np.float32) / 255.0
#         input_image = Image.fromarray((dummy_input * 255).astype(np.uint8))
#         print(f"Testing Core ML model with dummy input image...")
#         predictions = mlmodel.predict({"inputs": input_image})
#         print(f"Core ML model predictions: {predictions}")
#     except Exception as e:
#         print(f"Error testing Core ML model: {e}")


# def clear_resources(*resources):
#     """Clear memory resources."""
#     print("\n==== Cleaning Up Resources ====")
#     for resource in resources:
#         try:
#             del resource
#         except NameError:
#             print("Resource already cleared.")
#     tf.keras.backend.clear_session()
#     gc.collect()

#     try:
#         from numba import cuda
#         cuda.select_device(0)  # Adjust if using multiple GPUs
#         cuda.close()
#         print("GPU memory cleared.")
#     except ModuleNotFoundError:
#         print("Numba not found. Skipping GPU memory cleanup.")

#     print("Resources cleaned up.")


# # ========================
# # Main Script
# # ========================
# model = None
# mlmodel = None

# try:
#     # Load the trained Keras model
#     print("\n==== Loading TensorFlow Keras Model ====")
#     keras_model_path = "best_nsfw_safe_model.keras"
#     if not os.path.exists(keras_model_path):
#         raise FileNotFoundError(f"Model file not found: {keras_model_path}")
#     model = tf.keras.models.load_model(keras_model_path)
#     print(f"Successfully loaded model from {keras_model_path}")

#     # Test TensorFlow model predictions
#     log_tensorflow_model(model)

#     # Save the model in TensorFlow SavedModel format
#     saved_model_dir = "saved_model"
#     if os.path.exists(saved_model_dir):
#         print("Deleting existing SavedModel directory...")
#         import shutil
#         shutil.rmtree(saved_model_dir)

#     print("\n==== Saving Model in TensorFlow SavedModel Format ====")
#     tf.saved_model.save(
#         model,
#         saved_model_dir,
#         signatures=tf.function(model.call).get_concrete_function(
#             tf.TensorSpec([None, 512, 512, 3], tf.float32)
#         ),
#     )
#     print(f"Model successfully saved in TensorFlow SavedModel format: {saved_model_dir}")

#     # Convert the SavedModel to Core ML format
#     print("\n==== Converting to Core ML Format ====")
#     class_labels = ["nsfw", "safe"]
#     mlmodel = ct.convert(
#         saved_model_dir,
#         source="tensorflow",
#         inputs=[ct.ImageType(shape=(512, 512, 3), color_layout="RGB")],
#         classifier_config=ct.ClassifierConfig(class_labels),
#     )
#     print("Successfully converted model to Core ML format.")

#     # Save the Core ML model in both .mlmodel and .mlpackage formats
#     mlmodel.save("NSFWSafeClassifier.mlmodel")
#     mlmodel.save("NSFWSafeClassifier.mlpackage")
#     print("Model successfully converted to Core ML format and saved as 'NSFWSafeClassifier.mlpackage'!")

#     # Test Core ML model predictions
#     test_coreml_model(mlmodel)

# except FileNotFoundError as fnfe:
#     print(f"\nERROR: {fnfe}")

# except Exception as e:
#     print(f"\nERROR: {e}")

# finally:
#     # Cleanup resources
#     clear_resources(model, mlmodel)

# print("\n==== Script Finished ====")


# import coremltools as ct
# import tensorflow as tf
# import numpy as np
# import gc
# import os
# from PIL import Image


# def log_tensorflow_model(model, input_shape=(1, 512, 512, 3)):
#     print("\n==== TensorFlow Model Testing ====")
#     try:
#         dummy_input = np.random.rand(*input_shape).astype(np.float32)
#         output = model.predict(dummy_input)
#         print(f"Model output shape: {output.shape}, values: {output}")
#     except Exception as e:
#         print(f"Error testing TensorFlow model: {e}")


# def test_coreml_model(mlmodel, input_shape=(512, 512, 3)):
#     print("\n==== Core ML Model Testing ====")
#     try:
#         dummy_input = np.random.rand(*input_shape).astype(np.float32) / 255.0
#         input_image = Image.fromarray((dummy_input * 255).astype(np.uint8))
#         predictions = mlmodel.predict({"inputs": input_image})
#         print(f"Core ML model predictions: {predictions}")
#     except Exception as e:
#         print(f"Error testing Core ML model: {e}")


# def clear_resources(*resources):
#     for resource in resources:
#         try:
#             del resource
#         except NameError:
#             pass
#     tf.keras.backend.clear_session()
#     gc.collect()


# model = None
# mlmodel = None

# try:
#     keras_model_path = "best_nsfw_safe_model.keras"
#     if not os.path.exists(keras_model_path):
#         raise FileNotFoundError(f"Model file not found: {keras_model_path}")
#     model = tf.keras.models.load_model(keras_model_path)
#     print("Successfully loaded TensorFlow model.")

#     log_tensorflow_model(model)

#     saved_model_dir = "saved_model"
#     if os.path.exists(saved_model_dir):
#         import shutil
#         shutil.rmtree(saved_model_dir)

#     print("\nSaving TensorFlow model as SavedModel format...")
#     tf.saved_model.save(
#         model,
#         saved_model_dir,
#         signatures=tf.function(model.call).get_concrete_function(
#             tf.TensorSpec([None, 512, 512, 3], tf.float32)
#         ),
#     )

#     class_labels = ["nsfw", "safe"]
#     print("\nConverting to Core ML format...")
#     mlmodel = ct.convert(
#         saved_model_dir,
#         source="tensorflow",
#         inputs=[ct.ImageType(shape=(512, 512, 3), color_layout="RGB")],
#         classifier_config=ct.ClassifierConfig(class_labels),
#     )

#     mlmodel.save("NSFWSafeClassifier.mlpackage")
#     print("Model successfully saved as 'NSFWSafeClassifier.mlpackage'!")

#     test_coreml_model(mlmodel)

# except Exception as e:
#     print(f"\nERROR: {e}")

# finally:
#     clear_resources(model, mlmodel)

# print("\n==== Script Finished ====")


# import tensorflow as tf
# import json

# # Paths
# config_path = "extracted_model/config.json"
# weights_path = "extracted_model/model.weights.h5"
# output_model_path = "rebuilt_model.h5"

# def rebuild_model(config_path, weights_path, output_model_path):
#     try:
#         # Load the model configuration
#         with open(config_path, "r") as f:
#             model_config = json.load(f)
#         print("Model config loaded successfully.")
#     except Exception as e:
#         print(f"Error loading config.json: {e}")
#         return

#     # Validate and inspect configuration
#     if "config" not in model_config or "layers" not in model_config["config"]:
#         print("Invalid model configuration format: Missing 'config' or 'layers'.")
#         return

#     for i, layer in enumerate(model_config["config"]["layers"]):
#         print(f"Inspecting layer {i + 1}/{len(model_config['config']['layers'])}: {layer.get('class_name')}")
#         try:
#             # Debugging layer config
#             print(f"Layer {i + 1} full config: {layer}")

#             # Validate 'config' key in layer
#             if "config" not in layer or layer["config"] is None:
#                 print(f"Layer {i + 1} is missing 'config' or it is None. Skipping.")
#                 continue

#             # Process 'dtype'
#             if "dtype" in layer["config"]:
#                 dtype_config = layer["config"]["dtype"]
#                 if isinstance(dtype_config, dict) and dtype_config.get("class_name") == "DTypePolicy":
#                     layer["config"]["dtype"] = "float32"
#                     print(f"Updated dtype for layer {i + 1}.")

#             # Remove 'batch_shape'
#             if "batch_shape" in layer["config"]:
#                 del layer["config"]["batch_shape"]
#                 print(f"Removed 'batch_shape' for layer {i + 1}.")

#         except Exception as e:
#             print(f"Error processing layer {i + 1}: {e}")
#             return

#     # Attempt to rebuild the model
#     try:
#         with tf.keras.utils.custom_object_scope({"DTypePolicy": tf.keras.mixed_precision.Policy}):
#             model = tf.keras.Sequential.from_config(model_config["config"])
#         print("Model rebuilt successfully.")
#     except Exception as e:
#         print(f"Error rebuilding model: {e}")
#         return

#     # Load weights
#     try:
#         model.load_weights(weights_path)
#         print("Weights loaded successfully.")
#     except Exception as e:
#         print(f"Error loading weights: {e}")
#         return

#     # Save model
#     try:
#         model.save(output_model_path)
#         print(f"Model saved successfully at {output_model_path}.")
#     except Exception as e:
#         print(f"Error saving model: {e}")

# # Run the function
# rebuild_model(config_path, weights_path, output_model_path)


#good last one -----------------------------------------------------------------
# import coremltools as ct
# import tensorflow as tf
# import numpy as np
# import gc
# import os

# # ========================
# # Helper Functions
# # ========================

# def log_tensorflow_model(model, input_shape=(1, 512, 512, 3)):
#     """Test TensorFlow model predictions and log outputs."""
#     print("\n==== TensorFlow Model Testing ====")
#     try:
#         # Create a dummy input
#         dummy_input = np.random.rand(*input_shape).astype(np.float32)
#         print(f"Testing model with dummy input of shape {dummy_input.shape}...")
#         output = model.predict(dummy_input)
#         print(f"Model output shape: {output.shape}")
#         print(f"Model output values: {output}")

#         if output.shape[-1] == 2:
#             print(f"Detected 2-class output: Assuming [NSFW, SAFE]")
#         else:
#             print(f"Unexpected output shape: {output.shape}")
#     except Exception as e:
#         print(f"Error testing TensorFlow model: {e}")


# def test_coreml_model(mlmodel, input_shape=(1, 512, 512, 3)):
#     """Test Core ML model predictions and log outputs."""
#     print("\n==== Core ML Model Testing ====")
#     try:
#         # Core ML predict requires macOS 10.13 or later
#         if os.name != "posix":
#             print("Core ML model prediction is only supported on macOS.")
#             return
#         # Create a dummy input
#         dummy_input = np.random.rand(*input_shape).astype(np.float32)
#         print(f"Testing Core ML model with dummy input of shape {dummy_input.shape}...")
#         predictions = mlmodel.predict({"input": dummy_input})
#         print(f"Core ML model predictions: {predictions}")
#     except Exception as e:
#         print(f"Error testing Core ML model: {e}")


# def clear_resources(*resources):
#     """Clear memory resources."""
#     print("\n==== Cleaning Up Resources ====")
#     for resource in resources:
#         try:
#             del resource
#         except NameError:
#             print("Resource already cleared.")
#     tf.keras.backend.clear_session()
#     gc.collect()

#     try:
#         from numba import cuda
#         cuda.select_device(0)  # Adjust if using multiple GPUs
#         cuda.close()
#         print("GPU memory cleared.")
#     except ModuleNotFoundError:
#         print("Numba not found. Skipping GPU memory cleanup.")

#     print("Resources cleaned up.")


# # ========================
# # Main Script
# # ========================
# model = None
# mlmodel = None

# try:
#     # Load the trained Keras model
#     print("\n==== Loading TensorFlow Keras Model ====")
#     keras_model_path = "best_nsfw_safe_model.keras"
#     if not os.path.exists(keras_model_path):
#         raise FileNotFoundError(f"Model file not found: {keras_model_path}")
#     model = tf.keras.models.load_model(keras_model_path)
#     print(f"Successfully loaded model from {keras_model_path}")

#     # Test TensorFlow model predictions
#     log_tensorflow_model(model)

#     # Save the model in TensorFlow SavedModel format
#     saved_model_dir = "saved_model"
#     print("\n==== Saving Model in TensorFlow SavedModel Format ====")
#     tf.saved_model.save(
#         model,
#         saved_model_dir,
#         signatures=tf.function(model.call).get_concrete_function(
#             tf.TensorSpec([None, 512, 512, 3], tf.float32)
#         ),
#     )
#     print(f"Model successfully saved in TensorFlow SavedModel format: {saved_model_dir}")

#     # Convert the SavedModel to Core ML format
#     print("\n==== Converting to Core ML Format ====")
#     class_labels = ["nsfw", "safe"]
#     mlmodel = ct.convert(
#         saved_model_dir,
#         source="tensorflow",
#         inputs=[ct.ImageType(shape=(1, 512, 512, 3))],
#         classifier_config=ct.ClassifierConfig(class_labels),
#         convert_to="neuralnetwork",  # Ensures .mlmodel format
#     )
#     print("Successfully converted model to Core ML format.")

#     # Save the Core ML model in .mlmodel format
#     # coreml_model_path = "NSFWSafeClassifier.mlmodel"
#     # mlmodel.save(coreml_model_path)
#     # print(f"Core ML model saved as: {coreml_model_path}")

#     mlmodel.save("NSFWSafeClassifier.mlpackage")
#     print("Model successfully converted to Core ML format and saved as 'NSFWSafeClassifier.mlpackage'!")

#     # Test Core ML model predictions
#     test_coreml_model(mlmodel)

# except FileNotFoundError as fnfe:
#     print(f"\nERROR: {fnfe}")

# except Exception as e:
#     print(f"\nERROR: {e}")

# finally:
#     # Cleanup resources
#     clear_resources(model, mlmodel)

# print("\n==== Script Finished ====")


#-------------------------------------------------------------------------------


# import tensorflow as tf
# print(tf.reduce_sum(tf.random.normal([1000, 1000])))


import coremltools as ct
import tensorflow as tf
import numpy as np
import gc
import os
from PIL import Image
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model

# ========================
# Helper Functions
# ========================

def log_tensorflow_model(model, input_shape=(1, 512, 512, 3)):
    """Test TensorFlow model predictions and log outputs."""
    print("\n==== TensorFlow Model Testing ====")
    try:
        # Create a dummy input
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        print(f"Testing model with dummy input of shape {dummy_input.shape}...")
        output = model.predict(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Model output values: {output}")

        if output.shape[-1] == 2:
            print(f"Detected 2-class output: Assuming [nsfw, safe]")
        else:
            print(f"Unexpected output shape: {output.shape}")
    except Exception as e:
        print(f"Error testing TensorFlow model: {e}")

def test_coreml_model(mlmodel, image_path="1.jpg"):
    """Test Core ML model predictions and log outputs."""
    print("\n==== Core ML Model Testing ====")
    try:
        # Open and preprocess image
        print(f"Opening image: {image_path}...")
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))  # Ensure the image is 512x512

        print(f"Testing Core ML model with image of shape {image.size}...")

        # Prepare input dictionary for Core ML
        input_dict = {'inputs': image}  # Pass the PIL.Image directly

        # Run the model
        print("Running the model...")
        predictions = mlmodel.predict(input_dict)

        # Extract predictions correctly
        class_label = predictions.get("classLabel", "Unknown")
        probabilities = predictions.get("classLabel_probs", {})

        # Print the results
        print("Predictions:")
        print(f"Class Label: {class_label}")
        for label, confidence in probabilities.items():
            print(f"Label: {label}, Confidence: {confidence}")

    except Exception as e:
        print(f"Error testing Core ML model: {e}")



def clear_resources(*resources):
    """Clear memory resources."""
    print("\n==== Cleaning Up Resources ====")
    for resource in resources:
        try:
            del resource
        except NameError:
            print("Resource already cleared.")
    tf.keras.backend.clear_session()
    gc.collect()

    print("Resources cleaned up.")

def sequential_to_functional(sequential_model):
    """Convert a Sequential model to Functional."""
    inputs = Input(shape=sequential_model.input_shape[1:])
    outputs = sequential_model(inputs)
    return Model(inputs=inputs, outputs=outputs)

# def ensure_tracked(model):
#     """Ensure TensorFlow variables are properly tracked."""
#     for layer in model.layers:
#         if hasattr(layer, 'kernel') and layer.kernel is not None:
#             layer.add_weight(
#                 name='kernel',
#                 shape=layer.kernel.shape,
#                 dtype=tf.float32,
#                 initializer=tf.keras.initializers.Constant(layer.kernel.numpy())
#             )

# ========================
# Main Script
# ========================
model = None
mlmodel = None


try:
    print("\n==== Loading TensorFlow Keras Model ====")
    keras_model_path = "best_nsfw_safe_model.keras"
    h5_model_path = "model.h5"
    saved_model_dir = "saved_model"

    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Model file not found: {keras_model_path}")

    model = tf.keras.models.load_model(keras_model_path)
    print(f"Successfully loaded model from {keras_model_path}")

    # Ensure all variables are in float32 precision
    # ensure_tracked(model)
    for layer in model.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            layer.kernel.assign(tf.cast(layer.kernel, tf.float32))
    for layer in model.layers:
        if hasattr(layer, 'dtype') and layer.dtype == 'float16':
            layer.dtype = 'float32'

    # Convert to Functional Model
    model = sequential_to_functional(model)
    print("\nModel Summary:")
    model.summary()

    # Save the model in HDF5 format for additional checks
    model.save(h5_model_path)
    h5_model = load_model(h5_model_path)
    h5_model.summary()

    # Save in TensorFlow SavedModel format
    print("\n==== Saving Model in TensorFlow SavedModel Format ====")
    tf.saved_model.save(
        h5_model,
        saved_model_dir,
        signatures=tf.function(h5_model.call).get_concrete_function(
            tf.TensorSpec([None, 512, 512, 3], tf.float32)
        ),
    )
    print(f"Model successfully saved in TensorFlow SavedModel format: {saved_model_dir}")

    # Convert to Core ML
    print("\n==== Converting to Core ML Format ====")
    class_labels = ["nsfw", "safe"]
    mlmodel = ct.convert(
        saved_model_dir,
        source="tensorflow",
        inputs=[ct.ImageType(shape=(1, 512, 512, 3), scale=1/255.0)],
        classifier_config=ct.ClassifierConfig(class_labels),
        convert_to="mlprogram"
    )
    print("Successfully converted model to Core ML format.")

    # Save the Core ML model
    # mlmodel.save("NSFWSafeClassifier.mlmodel")
    # print(f"Core ML model saved as: 'NSFWSafeClassifier.mlmodel'")

    mlmodel.save("NSFWSafeClassifier.mlpackage")
    print("Model successfully converted to Core ML format and saved as 'NSFWSafeClassifier.mlpackage'!")

    # Test the Core ML model
    test_coreml_model(mlmodel)

except FileNotFoundError as fnfe:
    print(f"\nERROR: {fnfe}")

except Exception as e:
    print(f"\nERROR: {e}")

finally:
    clear_resources(model, mlmodel)

print("\n==== Script Finished ====")












# tmux new -s convert_to_coreml
# python3 convert_to_coreml.py 2>&1 | tee output3.log
# Detach: Press Ctrl+b followed by d.
# Reattach: tmux attach -t convert_to_coreml
# tail -f output3.log
# tmux ls
# tmux kill-session -t convert_to_coreml

#run after in terminal: sudo sync && sudo sysctl -w vm.drop_caches=3
# To see gpu usage: watch -n 1 nvidia-smi
# To see memory usage: watch -n 1 free -h
# htop
# cat output3.log | tee temp3.txt
# rm output3.log

#otherwise to run it: python3 convert_to_coreml.py

#To transfer the mlpackage folder from remote desktop to mac in the current directory:
#scp -r wsl:/home/lagoupo/code/ldlefebvre/explicit-model/model_two_categories/NSFWSafeClassifier.mlpackage .

#docker build --platform=linux/arm64 -t coreml_converter .
#docker run --rm -it --platform=linux/amd64 coreml_converter /bin/bash

# /Users/ldlefebvre/miniforge3/bin/conda init zsh
# source ~/.zshrc
# conda activate base
# conda install -c conda-forge coremltools

#dict issue:
#pip install wrapt==1.14.1
