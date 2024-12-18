import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import set_global_policy

# ========================
# Enable Mixed Precision
# ========================
set_global_policy('mixed_float16')  # Use 16-bit precision for faster training and lower memory usage

# ========================
# Dataset Directories
# ========================
TRAIN_DIR = "dataset/train"  # Path to training dataset
TEST_DIR = "dataset/test"    # Path to testing dataset
MODEL_NAME = "nsfw_safe_model.h5"  # Saved model name
BEST_MODEL_NAME = "best_nsfw_safe_model.h5"  # Best model during training

# ========================
# Hyperparameters
# ========================
BATCH_SIZE = 32
EPOCHS = 20  # Max epochs (early stopping will stop earlier)
DROPOUT_RATE = 0.5  # Dropout rate to reduce overfitting

# ========================
# Data Augmentation and Preprocessing
# ========================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,            # Normalize pixel values to [0, 1]
    horizontal_flip=True,         # Random horizontal flip
    rotation_range=15,            # Rotate images by up to 15 degrees
    zoom_range=0.2,               # Random zoom
    brightness_range=[0.8, 1.2],  # Random brightness
    width_shift_range=0.2,        # Horizontal shift
    height_shift_range=0.2        # Vertical shift
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Normalize test data

# ========================
# Load Training and Testing Data
# ========================
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=None,  # Use original image size
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=None,  # Use original image size
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ========================
# Define CNN Model
# ========================
input_shape = train_data.image_shape  # Dynamically get the input shape

model = Sequential([
    # Convolutional layers
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(DROPOUT_RATE),  # Dropout to reduce overfitting
    Dense(train_data.num_classes, activation='softmax', dtype='float32')  # Output layer
])

# ========================
# Compile the Model
# ========================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ========================
# Callbacks: Early Stopping and Model Checkpoints
# ========================
early_stop = EarlyStopping(
    monitor='val_loss',        # Monitor validation loss
    patience=3,                # Stop after 3 epochs with no improvement
    restore_best_weights=True  # Restore the best weights
)

checkpoint = ModelCheckpoint(
    BEST_MODEL_NAME,           # File to save the best model
    monitor='val_loss',        # Save the model with the lowest validation loss
    save_best_only=True,       # Only save when there's improvement
    mode='min',
    verbose=1
)

# ========================
# Train the Model
# ========================
print("Starting model training...")
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ========================
# Save the Final Model
# ========================
model.save(MODEL_NAME)
print(f"Training complete! Final model saved as {MODEL_NAME}")

# ========================
# Evaluate on Test Data
# ========================
print("Evaluating model on test data...")
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


#to run it: python train_model.py
