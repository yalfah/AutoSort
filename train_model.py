import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'dataset'
MODEL_SAVE_PATH = 'trash_classifier.tflite'

def build_and_train():
    # 1. Data Preparation with Augmentation
    # Augmentation helps the model generalize by rotating/zooming images artificially
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Use 20% of data for validation
    )

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Binary: Trash vs Recycle
        subset='training'
    )

    print("Loading Validation Data...")
    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    # 2. Transfer Learning Setup (MobileNetV2)
    # include_top=False removes the final classification layer of MobileNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze base model layers so we don't destroy pre-trained patterns
    base_model.trainable = False

    # 3. Add Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)  # Prevent overfitting
    predictions = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification

    model = Model(inputs=base_model.input, outputs=predictions)

    # 4. Compile and Train
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("Starting Training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # 5. Convert to TensorFlow Lite (Edge Optimization)
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Quantization (makes model 4x smaller, slightly less accurate)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] 
    
    tflite_model = converter.convert()

    # Save the file
    with open(MODEL_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Success! Model saved to {MODEL_SAVE_PATH}")
    print(f"Class Indices: {train_generator.class_indices}")

if __name__ == "__main__":
    # check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' directory not found. Please create it and add 'trash' and 'recycle' subfolders.")
    else:
        build_and_train()