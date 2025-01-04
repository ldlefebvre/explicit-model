import os
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import gc
import math
from tqdm import tqdm
from PIL import Image
import csv

# Enable Mixed Precision
set_global_policy('mixed_float16')

# Dataset Directories
TEST_DIR = "dataset/test"
BEST_MODEL_NAME = "best_nsfw_safe_model.keras"

# Hyperparameters
BATCH_SIZE = 32

# Data Augmentation for Testing
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load Test Data
test_data = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, test_data.num_classes), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

# Calculate Validation Steps
val_steps = math.ceil(test_data.samples / BATCH_SIZE)

# Reload Best Model
print("Loading the best model...")
model = load_model(BEST_MODEL_NAME)

# Evaluate Model
# print("Evaluating the model...")
# test_loss, test_acc = model.evaluate(test_data, steps=val_steps)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")
# print(f"Test Loss: {test_loss:.4f}")

# Function to Calculate Per-Class Accuracy
def calculate_per_class_accuracy(model, data, class_indices):
    y_true = []
    y_pred = []

    # Get the total number of batches in the data
    total_batches = len(data)
    print(f"Total number of batches: {total_batches}")

    # Process data batch by batch
    for batch_idx in range(total_batches):
        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        batch_images, batch_labels = next(data)  # Use Python's `next` to get the batch
        batch_predictions = model.predict(batch_images, verbose=1)
        y_pred.extend(np.argmax(batch_predictions, axis=-1))
        y_true.extend(np.argmax(batch_labels, axis=-1))

    # Map indices to class names
    class_names = {v: k for k, v in class_indices.items()}
    y_true_named = [class_names[idx] for idx in y_true]
    y_pred_named = [class_names[idx] for idx in y_pred]

    # Print classification report
    report = classification_report(y_true_named, y_pred_named, target_names=list(class_names.values()), zero_division=0)
    print("\nPer-Class Accuracy:\n", report)


# Function to Plot and Save Confusion Matrix
def plot_confusion_matrix(model, data, class_indices):
    y_true = []
    y_pred = []

    # Calculate the total number of samples
    total_samples = data.samples  # Total number of images in the dataset
    print(f"Total number of samples in the dataset: {total_samples}")

    # Process the data in batches
    for batch_idx in range(len(data)):
        batch_images, batch_labels = next(data)  # Use `next()` to get the next batch
        print(f"Processing batch {batch_idx + 1}/{len(data)}...")

        # Make predictions
        batch_predictions = model.predict(batch_images, verbose=1)
        y_pred.extend(np.argmax(batch_predictions, axis=-1))  # Predicted class indices
        y_true.extend(np.argmax(batch_labels, axis=-1))       # True class indices

    # Map indices to class names
    class_names = {v: k for k, v in class_indices.items()}  # Reverse the mapping of class_indices
    y_true_named = [class_names[idx] for idx in y_true]
    y_pred_named = [class_names[idx] for idx in y_pred]

    # Generate confusion matrix with class names
    cm = confusion_matrix(y_true_named, y_pred_named, labels=list(class_names.values()))

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_names.values()))
    disp.plot(cmap='viridis', xticks_rotation=45)
    plt.title("Confusion Matrix")

    # Save the confusion matrix plot
    output_path = "confusion_matrix_with_names.png"  # Path to save the plot
    plt.savefig(output_path, bbox_inches='tight')  # Save with a tight layout
    plt.close()  # Close the plot to release resources
    print(f"Confusion matrix saved to {output_path}")

# Per-Class Accuracy
# calculate_per_class_accuracy(model, test_data, test_data.class_indices)

# Confusion Matrix
# plot_confusion_matrix(model, test_data, test_data.class_indices)

# Function to Collect Misclassified Images
def get_misclassified_images(model, dataset, class_indices, filenames,
                             batch_size=1,
                             problematic_byte_size=100663295):
    misclassified_images = []
    class_names = {v: k for k, v in class_indices.items()}
    print("Processing dataset element by element...")

    if hasattr(dataset, "reset"):
        dataset.reset()

    total_samples = dataset.samples if hasattr(dataset, "samples") else len(dataset)
    print(f"Total samples in dataset: {total_samples}")

    num_batches = math.ceil(total_samples / batch_size)

    processed_so_far = 0

    with tqdm(total=total_samples, desc="Processing Images", unit="img") as progress_bar:
    # with tqdm(total=total_samples, desc="Processing Images") as progress_bar:
        for batch_idx in range(num_batches):
            batch_images, batch_labels = next(test_data)
            current_batch_len = len(batch_images)

            # If adding current_batch_len would exceed total_samples, clamp it
            if processed_so_far + current_batch_len > total_samples:
                current_batch_len = total_samples - processed_so_far

        # for batch_idx, (batch_images, batch_labels) in enumerate(dataset):
        #     current_batch_len = len(batch_images)  # store length first
            # Instead of calling model.predict() for each image, do it once:
            try:
                batch_predictions = model.predict(batch_images, verbose=0)
                for i in range(len(batch_images)):
                    # Optional: skip huge images
                    if isinstance(batch_images[i], np.ndarray):
                        if batch_images[i].nbytes > problematic_byte_size:
                            print(f"Skipping large image at batch {batch_idx}, index {i}")
                            continue

                    true_label = np.argmax(batch_labels[i])
                    predicted_label = np.argmax(batch_predictions[i])

                    if predicted_label != true_label:
                        # Only store filename + predicted/true
                        misclassified_images.append(
                            (filenames[batch_idx * batch_size + i],
                             class_names[predicted_label],
                             class_names[true_label])
                        )

                # Clear big arrays
                del batch_images, batch_labels, batch_predictions
                gc.collect()

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

            finally:
                # Now you can update the progress bar safely
                progress_bar.update(current_batch_len)
                processed_so_far += current_batch_len

            # If we've reached total_samples exactly, break out of the loop
            if processed_so_far >= total_samples:
                break

    print(f"Total misclassified images: {len(misclassified_images)}")
    return misclassified_images

# Function to Visualize Misclassified Images
# def visualize_misclassified_images(misclassified_images, output_path="misclassified_images.png", max_images=50):
#     total_images = len(misclassified_images)
#     print(f"Total Misclassified Images: {total_images}")

#     if total_images == 0:
#         print("No misclassified images to display.")
#         return

#     # Limit the number of images if necessary
#     # if total_images > max_images:
#     #     print(f"Limiting to first {max_images} images for display.")
#     #     misclassified_images = misclassified_images[:max_images]

#     cols = 5
#     rows = (len(misclassified_images) // cols) + (len(misclassified_images) % cols > 0)

#     fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
#     axes = axes.flatten()

#     # for i, (img, pred_label, true_label, file_name) in enumerate(misclassified_images):
#     #     axes[i].imshow(img)
#     #     axes[i].axis('off')
#     #     axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}\nFile: {file_name}", fontsize=10)

#     for i, (file_name, pred_label, true_label) in enumerate(misclassified_images):
#         # Build the full path: "dataset/test/<subdir>/<filename>"
#         # img_path = os.path.join(TEST_DIR, file_name)

#         # Load the actual image from disk
#         # img = Image.open(img_path).convert("RGB")

#         # axes[i].imshow(img)
#         axes[i].axis('off')
#         axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}\nFile: {file_name}", fontsize=10)

#     for j in range(len(misclassified_images), len(axes)):
#         axes[j].axis('off')

#     plt.tight_layout()
#     print(f"Saving misclassified images to {output_path}")
#     plt.savefig(output_path, bbox_inches='tight', dpi=100)  # Save instead of showing
#     plt.close()  # Close the figure

def visualize_misclassified_images_as_csv(misclassified_images, output_path="misclassified_images.csv"):
    """
    Save a list of misclassified samples to a CSV file instead of visualizing them in a single image.

    Each element in `misclassified_images` should be a tuple of:
       (file_name, pred_label, true_label).
    """
    total_images = len(misclassified_images)
    print(f"Total Misclassified Images: {total_images}")

    if total_images == 0:
        print("No misclassified images to display or save.")
        return

    # Write the data to a CSV file
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Header row
        writer.writerow(["file_name", "pred_label", "true_label"])

        # Write each misclassification as a row
        for file_name, pred_label, true_label in misclassified_images:
            writer.writerow([file_name, pred_label, true_label])

    print(f"Saved misclassified samples to {output_path}")

# Collect and Visualize Misclassified Images
misclassified_images = get_misclassified_images(model, test_data, test_data.class_indices, test_data.filenames)
visualize_misclassified_images_as_csv(misclassified_images)

# Final Garbage Collection
gc.collect()

# tmux new -s restart_after_model_fit
# python3 restart_after_model_fit.py 2>&1 | tee output2.log
# Detach: Press Ctrl+b followed by d.
# Reattach: tmux attach -t restart_after_model_fit
# tail -f output2.log
# tmux ls
# tmux kill-session -t restart_after_model_fit

# rm output2.log
# To clear caches: sudo sync && sudo sysctl -w vm.drop_caches=3
# cat output2.log | tee temp2.txt
