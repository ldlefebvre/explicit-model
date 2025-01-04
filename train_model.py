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



# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.mixed_precision import set_global_policy
# from tensorflow.keras import Input
# from PIL import Image
# import gc
# import matplotlib.pyplot as plt
# import numpy as np

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

# def adjust_random_contrast(image):
#     contrast_factor = np.random.uniform(0.8, 1.2)
#     return tf.image.adjust_contrast(image, contrast_factor)

# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,             # Normalize pixel values to [0, 1]
#     horizontal_flip=True,          # Flip horizontally
#     rotation_range=15,             # Rotate images within [-15, 15] degrees
#     zoom_range=(0.8, 1.2),         # Randomly zoom in/out
#     brightness_range=[0.8, 1.2],   # Adjust brightness
#     channel_shift_range=0.1,       # Shift RGB channels
#     width_shift_range=0.2,         # Shift image horizontally by up to 20%
#     height_shift_range=0.2,        # Shift image vertically by up to 20%
#     preprocessing_function=adjust_random_contrast  # Random contrast adjustment
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
# # input_shape = train_data.image_shape
# model = Sequential([
#     Input(shape=(512, 512, 3)),
#     Conv2D(32, (3, 3), activation='relu'),
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

# def get_misclassified_images(model, dataset, class_indices, filenames):
#     misclassified_images = []
#     class_names = {v: k for k, v in class_indices.items()}  # Map class indices to names

#     for i, (images, labels) in enumerate(dataset.unbatch()):  # Process all images
#         predictions = model.predict(tf.expand_dims(images, axis=0), verbose=0)
#         predicted_label = tf.argmax(predictions, axis=-1).numpy()
#         true_label = tf.argmax(labels, axis=-1).numpy()
#         file_name = filenames[i]  # Retrieve file name from filenames list

#         if predicted_label != true_label:
#             misclassified_images.append((
#                 images.numpy(),                 # Image
#                 class_names[predicted_label[0]],  # Predicted label
#                 class_names[true_label[0]],       # True label
#                 file_name                         # File name
#             ))

#     # Sort misclassified images by their true label for better organization
#     misclassified_images.sort(key=lambda x: x[2])  # Sort by true label
#     return misclassified_images

# # Function to Visualize Misclassified Images
# def visualize_misclassified_images(misclassified_images):
#     total_images = len(misclassified_images)
#     print(f"Total Misclassified Images: {total_images}")

#     # Dynamically create grid
#     cols = 5
#     rows = total_images // cols + (total_images % cols > 0)

#     fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
#     axes = axes.flatten()

#     for i, (img, pred_label, true_label, file_name) in enumerate(misclassified_images):
#         axes[i].imshow(img)
#         axes[i].axis('off')
#         axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}\nFile: {file_name}", fontsize=10)

#     # Turn off remaining axes if total_images < rows * cols
#     for j in range(total_images, len(axes)):
#         axes[j].axis('off')

#     plt.tight_layout()
#     plt.show()

# # Collect and Visualize Misclassified Images
# misclassified_images = get_misclassified_images(model, test_dataset, train_data.class_indices, test_data.filenames)
# visualize_misclassified_images(misclassified_images)

# # Final Garbage Collection
# gc.collect()

# # #to run it: python3 train_model.py
# # #run after in terminal: sudo sync && sudo sysctl -w vm.drop_caches=3


# #working for full model
# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.mixed_precision import set_global_policy
# from tensorflow.keras import Input
# from PIL import Image
# import gc
# import matplotlib.pyplot as plt
# import numpy as np

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
# def adjust_random_contrast(image):
#     contrast_factor = np.random.uniform(0.8, 1.2)
#     return tf.image.adjust_contrast(image, contrast_factor)

# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255,             # Normalize pixel values to [0, 1]
#     horizontal_flip=True,          # Flip horizontally
#     rotation_range=15,             # Rotate images within [-15, 15] degrees
#     zoom_range=(0.8, 1.2),         # Randomly zoom in/out
#     brightness_range=[0.8, 1.2],   # Adjust brightness
#     channel_shift_range=0.1,       # Shift RGB channels
#     width_shift_range=0.2,         # Shift image horizontally by up to 20%
#     height_shift_range=0.2,        # Shift image vertically by up to 20%
#     preprocessing_function=adjust_random_contrast  # Random contrast adjustment
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
# ).prefetch(tf.data.AUTOTUNE)

# test_dataset = tf.data.Dataset.from_generator(
#     lambda: test_data,
#     output_signature=(
#         tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
#         tf.TensorSpec(shape=(None, test_data.num_classes), dtype=tf.float32)
#     )
# ).prefetch(tf.data.AUTOTUNE)

# # Define Model
# model = Sequential([
#     Input(shape=(512, 512, 3)),    # Explicitly define the input shape
#     Conv2D(32, (3, 3), activation='relu'),
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

# # Function to Collect Misclassified Images
# def get_misclassified_images(model, dataset, class_indices, filenames):
#     misclassified_images = []
#     class_names = {v: k for k, v in class_indices.items()}  # Map class indices to names

#     for i, (images, labels) in enumerate(dataset.unbatch()):  # Process all images
#         predictions = model.predict(tf.expand_dims(images, axis=0), verbose=0)
#         predicted_label = tf.argmax(predictions, axis=-1).numpy()
#         true_label = tf.argmax(labels, axis=-1).numpy()
#         file_name = filenames[i]  # Retrieve file name from filenames list

#         if predicted_label != true_label:
#             misclassified_images.append((images.numpy(), class_names[predicted_label[0]], class_names[true_label[0]], file_name))

#     misclassified_images.sort(key=lambda x: x[2])  # Sort by true label
#     return misclassified_images

# # Function to Visualize Misclassified Images
# def visualize_misclassified_images(misclassified_images):
#     total_images = len(misclassified_images)
#     print(f"Total Misclassified Images: {total_images}")

#     cols = 5
#     rows = total_images // cols + (total_images % cols > 0)

#     fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
#     axes = axes.flatten()

#     for i, (img, pred_label, true_label, file_name) in enumerate(misclassified_images):
#         axes[i].imshow(img)
#         axes[i].axis('off')
#         axes[i].set_title(f"Pred: {pred_label}\nTrue: {true_label}\nFile: {file_name}", fontsize=10)

#     for j in range(total_images, len(axes)):
#         axes[j].axis('off')

#     plt.tight_layout()
#     plt.show()

# # Collect and Visualize Misclassified Images
# misclassified_images = get_misclassified_images(model, test_dataset, train_data.class_indices, test_data.filenames)
# visualize_misclassified_images(misclassified_images)

# # Final Garbage Collection
# gc.collect()

# # To run: python3 train_model.py
# # To clear caches: sudo sync && sudo sysctl -w vm.drop_caches=3



import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy, LossScaleOptimizer
from tensorflow.keras import Input
from PIL import Image
import gc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import math
from tqdm import tqdm
from PIL import Image
import csv

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
def adjust_random_contrast(image):
    contrast_factor = np.random.uniform(0.8, 1.2)
    return tf.image.adjust_contrast(image, contrast_factor)

#shuffle=False to avoid infinite batches being created
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,             # Normalize pixel values to [0, 1]
    horizontal_flip=True,          # Flip horizontally
    rotation_range=15,             # Rotate images within [-15, 15] degrees
    zoom_range=(0.8, 1.2),         # Randomly zoom in/out
    brightness_range=[0.8, 1.2],   # Adjust brightness
    channel_shift_range=0.1,       # Shift RGB channels
    width_shift_range=0.2,         # Shift image horizontally by up to 20%
    height_shift_range=0.2,        # Shift image vertically by up to 20%
    preprocessing_function=adjust_random_contrast  # Random contrast adjustment
)

#shuffle=False to avoid infinite batches being created
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
).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_data,
    output_signature=(
        tf.TensorSpec(shape=(None, 512, 512, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, test_data.num_classes), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

# Define Model
model = Sequential([
    Input(shape=(512, 512, 3)),    # Explicitly define the input shape
    Conv2D(32, (3, 3), activation='relu'),
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
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
base_optimizer = tf.keras.optimizers.Adam()
optimizer = LossScaleOptimizer(base_optimizer)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(BEST_MODEL_NAME, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# Train Model
# history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks=[early_stop, checkpoint, lr_scheduler])
train_steps = math.ceil(train_data.samples / BATCH_SIZE)
val_steps   = math.ceil(test_data.samples  / BATCH_SIZE)

print("Train samples:", train_data.samples, "=> steps_per_epoch =", train_steps)
print("Val samples:  ", test_data.samples,  "=> validation_steps =", val_steps)

#Train model (fit)
history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps,  # <--- crucial for having real epoch boundaries
    validation_data=test_dataset,
    validation_steps=val_steps,   # <--- so validation also ends properly
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, lr_scheduler]
)

print("Training complete. Saving model...")

# Save Model
model.save(MODEL_NAME)

print("Model saved. Rest here is optional to verify the details of the model...")

# Evaluate Model
# test_loss, test_acc = model.evaluate(test_dataset)
test_loss, test_acc = model.evaluate(test_dataset, steps=val_steps)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

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
calculate_per_class_accuracy(model, test_data, test_data.class_indices)

# Confusion Matrix
plot_confusion_matrix(model, test_data, test_data.class_indices)

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

# To run: python3 train_model.py
# To clear caches: sudo sync && sudo sysctl -w vm.drop_caches=3
# To see gpu usage: watch -n 1 nvidia-smi
# To see memory usage: watch -n 1 free -h
# htop
# cat output2.log | tee temp.txt

# Other useful commands:
# mkdir -p dataset2/{train,test}/{nsfw,safe} && cp -r dataset/train/{hentai,porn} dataset2/train/nsfw && cp -r dataset/train/{neutral,drawings,sexy} dataset2/train/safe && cp -r dataset/test/{hentai,porn} dataset2/test/nsfw && cp -r dataset/test/{neutral,drawings,sexy} dataset2/test/safe

# mkdir -p dataset2/{train,test}/{nsfw,safe} && \
# rsync -a --ignore-errors dataset/train/hentai/ dataset2/train/nsfw/ | pv -lep -s $(find dataset/train/hentai/ -type f | wc -l) > /dev/null 2>&1 & \
# rsync -a --ignore-errors dataset/train/porn/ dataset2/train/nsfw/ | pv -lep -s $(find dataset/train/porn/ -type f | wc -l) > /dev/null 2>&1 & \
# rsync -a --ignore-errors dataset/train/neutral/ dataset2/train/safe/ | pv -lep -s $(find dataset/train/neutral/ -type f | wc -l) > /dev/null 2>&1 & \
# rsync -a --ignore-errors dataset/train/drawings/ dataset2/train/safe/ | pv -lep -s $(find dataset/train/drawings/ -type f | wc -l) > /dev/null 2>&1 & \
# rsync -a --ignore-errors dataset/train/sexy/ dataset2/train/safe/ | pv -lep -s $(find dataset/train/sexy/ -type f | wc -l) > /dev/null 2>&1 & \
# rsync -a --ignore-errors dataset/test/hentai/ dataset2/test/nsfw/ | pv -lep -s $(find dataset/test/hentai/ -type f | wc -l) > /dev/null 2>&1 & \
# rsync -a --ignore-errors dataset/test/porn/ dataset2/test/nsfw/ | pv -lep -s $(find dataset/test/porn/ -type f | wc -l) > /dev/null 2>&1 & \
# rsync -a --ignore-errors dataset/test/neutral/ dataset2/test/safe/ | pv -lep -s $(find dataset/test/neutral/ -type f | wc -l) > /dev/null 2>&1 & \
# rsync -a --ignore-errors dataset/test/drawings/ dataset2/test/safe/ | pv -lep -s $(find dataset/test/drawings/ -type f | wc -l) > /dev/null 2>&1 & \
# rsync -a --ignore-errors dataset/test/sexy/ dataset2/test/safe/ | pv -lep -s $(find dataset/test/sexy/ -type f | wc -l) > /dev/null 2>&1 & \
# wait

# nohup python3 train_model.py > output.log 2>&1 &
# ps aux | grep train_model.py
# tail -f output.log
# reptyr 13067
# kill 13067

# tmux new -s train_model
# python3 train_model.py 2>&1 | tee output.log
# Detach: Press Ctrl+b followed by d.
# Reattach: tmux attach -t train_model
# tail -f output.log
# tmux ls
# tmux kill-session -t train_model

# scp wsl:/home/lagoupo/code/ldlefebvre/explicit-model/dataset/test/drawings/004f0b56b85378df6a31813cb08b676e812b6e4c7a424b422936bebb341405d2.jpg .
# open 004f0b56b85378df6a31813cb08b676e812b6e4c7a424b422936bebb341405d2.jpg




# Copy all images from remote desktop to mac in the misclassified_image.csv with:
# ssh -M -S ~/.ssh/ssh_mux_wsl -fN wsl
# -M: Enables master mode for connection sharing.
# -S ~/.ssh/ssh_mux_wsl: Specifies the control socket path.
# -f: Requests SSH to go to the background after authentication.
# -N: Indicates that no command will be executed on the remote system.

# ssh -S ~/.ssh/ssh_mux_wsl wsl "tail -n +2 /home/lagoupo/code/ldlefebvre/explicit-model/misclassified_images.csv" \
# | while IFS=, read -r file_name pred_label true_label; do
#     # Create the target directory locally if it doesn't exist
#     mkdir -p "${true_label}"

#     # Define the full remote path to the image
#     remote_path="/home/lagoupo/code/ldlefebvre/explicit-model/dataset/test/${file_name}"

#     # Use rsync with the control socket to copy the image
#     rsync -e "ssh -S ~/.ssh/ssh_mux_wsl" -av "wsl:${remote_path}" "${true_label}/"
# done

# ssh -S ~/.ssh/ssh_mux_wsl -O exit wsl


# OR


# Step 1: Edit Your SSH Config File
# Add the following to your ~/.ssh/config file:

# Host wsl
#     ControlMaster auto
#     ControlPath ~/.ssh/ssh_mux_%h_%p_%r
#     ControlPersist yes
# ControlMaster auto: Enables automatic connection sharing.
# ControlPath: Specifies the path for the control socket.
# ControlPersist yes: Keeps the master connection open in the background after the initial session is closed.

# ssh wsl "tail -n +2 /home/lagoupo/code/ldlefebvre/explicit-model/misclassified_images.csv" \
# | while IFS=, read -r file_name pred_label true_label; do
#     mkdir -p "${true_label}"
#     remote_path="/home/lagoupo/code/ldlefebvre/explicit-model/dataset/test/${file_name}"
#     rsync -av "wsl:${remote_path}" "${true_label}/"
# done
