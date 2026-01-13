import os
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
TRAIN_DIR = os.path.join('.\Documents\GitHub\AutoSort\dataset', 'train')
VAL_DIR = os.path.join('.\Documents\GitHub\AutoSort\dataset', 'val')
MODEL_SAVE_PATH = 'trash_classifier.tflite'

def build_and_train():
    # 1. Setup Generators
    
    # TRAINING GENERATOR (With Augmentation)
    # This creates artificial variations of your training data to prevent overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # VALIDATION GENERATOR (No Augmentation)
    # We only rescale. We do NOT rotate/shift validation data because we want 
    # to test on "real" images to see how the model performs.
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    print("Loading Validation Data...")
    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # 2. Transfer Learning Setup (MobileNetV2)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False

    # 3. Add Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 4. Compile and Train
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("Starting Training...")
    # The validation_data argument here automatically uses your val set
    # to evaluate performance at the end of every epoch.
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # 5. Convert to TFLite
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(MODEL_SAVE_PATH, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Success! Model saved to {MODEL_SAVE_PATH}")
    print(f"Class Indices: {train_generator.class_indices}")

if __name__ == "__main__":
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print("Error: Dataset directories not found.")
        print("Please run 'python split_data.py' first.")
    else:
        build_and_train()