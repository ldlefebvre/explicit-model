# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.mixed_precision import set_global_policy
# from PIL import Image

# # ========================
# # Enable Mixed Precision
# # ========================
# set_global_policy('mixed_float16')  # Use 16-bit precision for faster training and lower memory usage

# # ========================
# # Dataset Directories
# # ========================
# TRAIN_DIR = "dataset/train"  # Path to training dataset
# TEST_DIR = "dataset/test"    # Path to testing dataset
# MODEL_NAME = "nsfw_safe_model.keras"
# BEST_MODEL_NAME = "best_nsfw_safe_model.keras"

# # ========================
# # Hyperparameters
# # ========================
# BATCH_SIZE = 32
# EPOCHS = 20  # Max epochs (early stopping will stop earlier)
# DROPOUT_RATE = 0.5  # Dropout rate to reduce overfitting

# # ========================
# # Data Preprocessing Helper Functions
# # ========================
# # def safe_load_img(path):
# #     try:
# #         img = Image.open(path)
# #         img.verify()  # Ensure the image is valid
# #         return Image.open(path).resize((512, 512))  # Resize to target size
# #     except Exception as e:
# #         print(f"Skipping invalid image: {path} - {e}")
# #         return None

# def preprocess_directory(directory):
#     for root, _, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             try:
#                 # Open the image and verify its validity
#                 img = Image.open(file_path)
#                 img.verify()  # Ensure the file is a valid image
#                 img = Image.open(file_path)  # Reopen the image to work with it

#                 # Check for oversized images
#                 if img.size[0] * img.size[1] > 89_478_485:  # Pillow's decompression bomb limit
#                     print(f"Removing oversized image: {file_path} - {img.size} pixels")
#                     os.remove(file_path)
#                 else:
#                     # Forcefully load the image to ensure it's not truncated
#                     img.load()
#             except (OSError, Exception) as e:
#                 print(f"Removing invalid or truncated image: {file_path} - {e}")
#                 os.remove(file_path)  # Remove invalid or truncated files
#             finally:
#                 img.close()  # Free memory
#                 gc.collect()  # Force garbage collection


# # Preprocess both training and testing datasets
# print("Preprocessing datasets...")
# preprocess_directory(TRAIN_DIR)
# preprocess_directory(TEST_DIR)

# # ========================
# # Data Augmentation and Preprocessing
# # ========================
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,            # Normalize pixel values to [0, 1]
#     horizontal_flip=True,         # Random horizontal flip
#     rotation_range=15,            # Rotate images by up to 15 degrees
#     zoom_range=0.2,               # Random zoom
#     brightness_range=[0.8, 1.2],  # Random brightness
#     width_shift_range=0.2,        # Horizontal shift
#     height_shift_range=0.2        # Vertical shift
# )

# test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Normalize test data

# # ========================
# # Load Training and Testing Data
# # ========================
# train_data = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(512, 512),  # Resize images during loading
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# test_data = test_datagen.flow_from_directory(
#     TEST_DIR,
#     target_size=(512, 512),  # Resize images during loading
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# # ========================
# # Define CNN Model
# # ========================
# input_shape = train_data.image_shape  # Dynamically get the input shape

# model = Sequential([
#     # Convolutional layers
#     Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#     MaxPooling2D(2, 2),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(DROPOUT_RATE),  # Dropout to reduce overfitting
#     Dense(train_data.num_classes, activation='softmax', dtype='float32')  # Output layer
# ])

# # ========================
# # Compile the Model
# # ========================
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # ========================
# # Callbacks: Early Stopping and Model Checkpoints
# # ========================
# early_stop = EarlyStopping(
#     monitor='val_loss',        # Monitor validation loss
#     patience=3,                # Stop after 3 epochs with no improvement
#     restore_best_weights=True  # Restore the best weights
# )

# checkpoint = ModelCheckpoint(
#     BEST_MODEL_NAME,           # File to save the best model
#     monitor='val_loss',        # Save the model with the lowest validation loss
#     save_best_only=True,       # Only save when there's improvement
#     mode='min',
#     verbose=1
# )

# # ========================
# # Train the Model
# # ========================
# print("Starting model training...")
# history = model.fit(
#     train_data,
#     validation_data=test_data,
#     epochs=EPOCHS,
#     callbacks=[early_stop, checkpoint]
# )

# # ========================
# # Save the Final Model
# # ========================
# model.save(MODEL_NAME)
# print(f"Training complete! Final model saved as {MODEL_NAME}")

# # ========================
# # Evaluate on Test Data
# # ========================
# print("Evaluating model on test data...")
# test_loss, test_acc = model.evaluate(test_data)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")



# #to run it: python3 train_model.py
# #run after in terminal: sudo sync && sudo sysctl -w vm.drop_caches=3


# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.mixed_precision import set_global_policy
# from PIL import Image

# # ========================
# # Enable Mixed Precision
# # ========================
# set_global_policy('mixed_float16')  # Use 16-bit precision for faster training and lower memory usage

# # ========================
# # Dataset Directories
# # ========================
# TRAIN_DIR = "dataset/train"  # Path to training dataset
# TEST_DIR = "dataset/test"    # Path to testing dataset
# MODEL_NAME = "nsfw_safe_model.keras"
# BEST_MODEL_NAME = "best_nsfw_safe_model.keras"

# # ========================
# # Hyperparameters
# # ========================
# BATCH_SIZE = 32
# EPOCHS = 20  # Max epochs (early stopping will stop earlier)
# DROPOUT_RATE = 0.5  # Dropout rate to reduce overfitting

# # ========================
# # Data Preprocessing Helper Functions
# # ========================
# # def safe_load_img(path):
# #     try:
# #         img = Image.open(path)
# #         img.verify()  # Ensure the image is valid
# #         return Image.open(path).resize((512, 512))  # Resize to target size
# #     except Exception as e:
# #         print(f"Skipping invalid image: {path} - {e}")
# #         return None

# def preprocess_directory(directory):
#     for root, _, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             try:
#                 # Open the image and verify its validity
#                 img = Image.open(file_path)
#                 img.verify()  # Ensure the file is a valid image
#                 img = Image.open(file_path)  # Reopen the image to work with it

#                 # Check for oversized images
#                 if img.size[0] * img.size[1] > 89_478_485:  # Pillow's decompression bomb limit
#                     print(f"Removing oversized image: {file_path} - {img.size} pixels")
#                     os.remove(file_path)
#                 else:
#                     # Forcefully load the image to ensure it's not truncated
#                     img.load()
#             except (OSError, Exception) as e:
#                 print(f"Removing invalid or truncated image: {file_path} - {e}")
#                 os.remove(file_path)  # Remove invalid or truncated files
#             finally:
#                 img.close()  # Free memory
#                 gc.collect()  # Force garbage collection


# # Preprocess both training and testing datasets
# print("Preprocessing datasets...")
# preprocess_directory(TRAIN_DIR)
# preprocess_directory(TEST_DIR)

# # ========================
# # Data Augmentation and Preprocessing
# # ========================
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,            # Normalize pixel values to [0, 1]
#     horizontal_flip=True,         # Random horizontal flip
#     rotation_range=15,            # Rotate images by up to 15 degrees
#     zoom_range=0.2,               # Random zoom
#     brightness_range=[0.8, 1.2],  # Random brightness
#     width_shift_range=0.2,        # Horizontal shift
#     height_shift_range=0.2        # Vertical shift
# )

# test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Normalize test data

# # ========================
# # Load Training and Testing Data
# # ========================
# train_data = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=(512, 512),  # Resize images during loading
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# test_data = test_datagen.flow_from_directory(
#     TEST_DIR,
#     target_size=(512, 512),  # Resize images during loading
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )

# # Convert to tf.data.Dataset
# train_dataset = tf.data.Dataset.from_generator(
#     lambda: train_data,
#     output_signature=(
#         tf.TensorSpec(shape=(BATCH_SIZE, 512, 512, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(BATCH_SIZE, train_data.num_classes), dtype=tf.float32)
#     )
# ).prefetch(tf.data.AUTOTUNE)

# test_dataset = tf.data.Dataset.from_generator(
#     lambda: test_data,
#     output_signature=(
#         tf.TensorSpec(shape=(BATCH_SIZE, 512, 512, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(BATCH_SIZE, test_data.num_classes), dtype=tf.float32)
#     )
# ).prefetch(tf.data.AUTOTUNE)

# # ========================
# # Define CNN Model
# # ========================
# input_shape = train_data.image_shape  # Dynamically get the input shape

# model = Sequential([
#     # Convolutional layers
#     Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#     MaxPooling2D(2, 2),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(DROPOUT_RATE),  # Dropout to reduce overfitting
#     Dense(train_data.num_classes, activation='softmax', dtype='float32')  # Output layer
# ])

# # ========================
# # Compile the Model
# # ========================
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # ========================
# # Callbacks: Early Stopping and Model Checkpoints
# # ========================
# early_stop = EarlyStopping(
#     monitor='val_loss',        # Monitor validation loss
#     patience=3,                # Stop after 3 epochs with no improvement
#     restore_best_weights=True  # Restore the best weights
# )

# checkpoint = ModelCheckpoint(
#     BEST_MODEL_NAME,           # File to save the best model
#     monitor='val_loss',        # Save the model with the lowest validation loss
#     save_best_only=True,       # Only save when there's improvement
#     mode='min',
#     verbose=1
# )

# lr_scheduler = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=2,
#     min_lr=1e-6,
#     verbose=1
# )

# # ========================
# # Train the Model
# # ========================
# print("Starting model training...")
# history = model.fit(
#     train_dataset,
#     validation_data=test_dataset,
#     epochs=EPOCHS,
#     callbacks=[early_stop, checkpoint, lr_scheduler]
# )

# # ========================
# # Save the Final Model
# # ========================
# model.save(MODEL_NAME)
# print(f"Training complete! Final model saved as {MODEL_NAME}")

# # ========================
# # Evaluate on Test Data
# # ========================
# print("Evaluating model on test data...")
# test_loss, test_acc = model.evaluate(test_dataset)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")



# #to run it: python3 train_model.py
# #run after in terminal: sudo sync && sudo sysctl -w vm.drop_caches=3



# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.mixed_precision import set_global_policy
# from PIL import Image
# import gc

# # Enable Mixed Precision
# set_global_policy('mixed_float16')

# # Dataset Directories
# TRAIN_DIR = "dataset/train"
# TEST_DIR = "dataset/test"
# MODEL_NAME = "nsfw_safe_model.keras"
# BEST_MODEL_NAME = "best_nsfw_safe_model.keras"

# # Hyperparameters
# BATCH_SIZE = 32
# EPOCHS = 20
# DROPOUT_RATE = 0.5

# # Preprocess Directory Function
# def preprocess_directory(directory):
#     for root, _, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             try:
#                 img = Image.open(file_path)
#                 img.verify()
#                 img = Image.open(file_path)
#                 if img.size[0] * img.size[1] > 89_478_485:
#                     print(f"Removing oversized image: {file_path}")
#                     os.remove(file_path)
#                 else:
#                     img.load()
#             except Exception as e:
#                 print(f"Removing invalid image: {file_path} - {e}")
#                 os.remove(file_path)
#             finally:
#                 img.close()
#     gc.collect()  # Cleanup memory after preprocessing

# # Preprocess Datasets
# preprocess_directory(TRAIN_DIR)
# preprocess_directory(TEST_DIR)

# # Data Augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     horizontal_flip=True,
#     rotation_range=15,
#     zoom_range=0.2,
#     brightness_range=[0.8, 1.2],
#     contrast_range=[0.8, 1.2],    # Random contrast adjustments
#     channel_shift_range=0.1,      # Random RGB channel shifts
#     width_shift_range=0.2,
#     height_shift_range=0.2
# )

# test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# # Load Data
# train_data = train_datagen.flow_from_directory(
#     TRAIN_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, class_mode='categorical'
# )
# test_data = test_datagen.flow_from_directory(
#     TEST_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, class_mode='categorical'
# )

# # Convert to tf.data.Dataset
# train_dataset = tf.data.Dataset.from_generator(
#     lambda: train_data,
#     output_signature=(
#         tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, train_data.num_classes), dtype=tf.float32)
#     )
# ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# test_dataset = tf.data.Dataset.from_generator(
#     lambda: test_data,
#     output_signature=(
#         tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, test_data.num_classes), dtype=tf.float32)
#     )
# ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# # Define Model
# input_shape = train_data.image_shape
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(DROPOUT_RATE),
#     Dense(train_data.num_classes, activation='softmax', dtype='float32')
# ])

# # Compile Model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Callbacks
# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# checkpoint = ModelCheckpoint(BEST_MODEL_NAME, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# # Train Model
# history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks=[early_stop, checkpoint, lr_scheduler])

# # Save Model
# model.save(MODEL_NAME)

# # Evaluate Model
# test_loss, test_acc = model.evaluate(test_dataset)
# print(f"Test Accuracy: {test_acc * 100:.2f}%")

# # Final Garbage Collection
# gc.collect()

# # #to run it: python3 train_model.py
# # #run after in terminal: sudo sync && sudo sysctl -w vm.drop_caches=3



import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from PIL import Image
import gc
import matplotlib.pyplot as plt
import numpy as np

# Enable Mixed Precision
set_global_policy('mixed_float16')

# Dataset Directories
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
MODEL_NAME = "nsfw_safe_model.keras"
BEST_MODEL_NAME = "best_nsfw_safe_model.keras"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
DROPOUT_RATE = 0.5

# Preprocess Directory Function
def preprocess_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
                img = Image.open(file_path)
                if img.size[0] * img.size[1] > 89_478_485:
                    print(f"Removing oversized image: {file_path}")
                    os.remove(file_path)
                else:
                    img.load()
            except Exception as e:
                print(f"Removing invalid image: {file_path} - {e}")
                os.remove(file_path)
            finally:
                img.close()
    gc.collect()  # Cleanup memory after preprocessing

# Preprocess Datasets
preprocess_directory(TRAIN_DIR)
preprocess_directory(TEST_DIR)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    contrast_range=[0.8, 1.2],    # Random contrast adjustments
    channel_shift_range=0.1,      # Random RGB channel shifts
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load Data
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(512, 512), batch_size=BATCH_SIZE, class_mode='categorical'
)

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, train_data.num_classes), dtype=tf.float32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, test_data.num_classes), dtype=tf.float32)
    )
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Define Model
input_shape = train_data.image_shape
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(train_data.num_classes, activation='softmax', dtype='float32')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(BEST_MODEL_NAME, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# Train Model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks=[early_stop, checkpoint, lr_scheduler])

# Save Model
model.save(MODEL_NAME)

# Evaluate Model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

def get_misclassified_images(model, dataset, class_indices, filenames):
    misclassified_images = []
    class_names = {v: k for k, v in class_indices.items()}  # Map class indices to names

    for i, (images, labels) in enumerate(dataset.unbatch()):  # Process all images
        predictions = model.predict(tf.expand_dims(images, axis=0), verbose=0)
        predicted_label = tf.argmax(predictions, axis=-1).numpy()
        true_label = tf.argmax(labels, axis=-1).numpy()
        file_name = filenames[i]  # Retrieve file name from filenames list

        if predicted_label != true_label:
            misclassified_images.append((
                images.numpy(),                 # Image
                class_names[predicted_label[0]],  # Predicted label
                class_names[true_label[0]],       # True label
                file_name                         # File name
            ))

    # Sort misclassified images by their true label for better organization
    misclassified_images.sort(key=lambda x: x[2])  # Sort by true label
    return misclassified_images

# Function to Visualize Misclassified Images
def visualize_misclassified_images(misclassified_images):
    total_images = len(misclassified_images)
    print(f"Total Misclassified Images: {total_images}")

    # Dynamically create grid
    cols = 5
    rows = total_images // cols + (total_images % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
    axes = axes.flatten()

    for i, (img, pred_label, true_label, file_name) in enumerate(misclassified_images):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}\nFile: {file_name}", fontsize=10)

    # Turn off remaining axes if total_images < rows * cols
    for j in range(total_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Collect and Visualize Misclassified Images
misclassified_images = get_misclassified_images(model, test_dataset, train_data.class_indices, test_data.filenames)
visualize_misclassified_images(misclassified_images)

# Final Garbage Collection
gc.collect()

# #to run it: python3 train_model.py
# #run after in terminal: sudo sync && sudo sysctl -w vm.drop_caches=3
