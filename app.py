import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import time

# --- CONFIGURATION ---
MODEL_PATH = 'trash_classifier.tflite'
INPUT_SHAPE = (224, 224)
THRESHOLD = 0.5  # Confidence threshold

def load_interpreter(model_path):
    """Loads the TFLite interpreter (Edge-optimized engine)"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image_path):
    """Resizes and normalizes image to match training format"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(INPUT_SHAPE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def predict(interpreter, image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess
    input_data = preprocess_image(image_path)

    # Run Inference
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()

    # Decode Result
    # Based on training: 0 = Recycle, 1 = Trash (or vice versa, checked below)
    # Note: You must verify class indices from training output. 
    # Usually alphanumeric: 'recycle' -> 0, 'trash' -> 1
    
    prediction_score = output_data[0][0]
    
    # Logic: If training indices were {'recycle': 0, 'trash': 1}
    # Score close to 0 is Recycle, Score close to 1 is Trash
    
    result = "TRASH" if prediction_score > THRESHOLD else "RECYCLE"
    confidence = prediction_score if result == "TRASH" else (1 - prediction_score)

    return result, confidence, (end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Edge Trash Classifier')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    args = parser.parse_args()

    try:
        interpreter = load_interpreter(MODEL_PATH)
        label, conf, duration = predict(interpreter, args.image)
        
        print("-" * 30)
        print(f"Prediction:  {label}")
        print(f"Confidence:  {conf*100:.2f}%")
        print(f"Time taken:  {duration*1000:.2f} ms")
        print("-" * 30)
        
        if label == "TRASH":
            print("Action: Discard in general waste.")
        else:
            print("Action: Safe to recycle.")
            
    except Exception as e:
        print(f"Error: {e}")