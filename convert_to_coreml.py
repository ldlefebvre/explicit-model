import coremltools as ct
import tensorflow as tf
import gc
from numba import cuda

# Load the trained Keras model
print("Loading the trained Keras model...")
model = tf.keras.models.load_model("best_nsfw_safe_model.keras")

# Save the model in TensorFlow SavedModel format
saved_model_dir = "saved_model"
print("Saving the model in TensorFlow SavedModel format...")
tf.saved_model.save(
    model,
    saved_model_dir,
    signatures=tf.function(model.call).get_concrete_function(
        tf.TensorSpec([None, 512, 512, 3], tf.float32)
    ),
)
print("Model saved in TensorFlow SavedModel format.")

# Convert the SavedModel to Core ML format and ensure the old .mlmodel format
print("Converting SavedModel to Core ML format...")
mlmodel = ct.convert(
    saved_model_dir,
    source="tensorflow",
    inputs=[ct.ImageType(shape=(1, 512, 512, 3))],
    convert_to="neuralnetwork",  # Ensures .mlmodel format
)

# Save the Core ML model in .mlmodel format
# mlmodel.save("NSFWSafeClassifier.mlmodel")
mlmodel.save("NSFWSafeClassifier.mlpackage")
print("Model successfully converted to Core ML format and saved as 'NSFWSafeClassifier.mlmodel'!")

# ========================
# Cleanup resources
# ========================
print("Cleaning up resources...")
# Clear TensorFlow session to release memory
tf.keras.backend.clear_session()

# Explicitly delete large objects
del model
del mlmodel

# Force garbage collection
gc.collect()

# Try to reset GPU memory if numba is installed
try:
    from numba import cuda
    cuda.select_device(0)  # Adjust if using multiple GPUs
    cuda.close()
    print("GPU memory cleared.")
except ModuleNotFoundError:
    print("Numba not found. Skipping GPU memory cleanup.")

print("Resources cleaned up. Script finished.")

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
