import onnxruntime as ort
import numpy as np
import cv2
import os
from PIL import Image

# Global session variable
_session = None

def _get_session():
    global _session
    if _session is None:
        model_path = os.path.join(os.getcwd(), "model", "mnist-12.onnx")
        _session = ort.InferenceSession(model_path)
    return _session

def preprocess_image(image_path):
    """Preprocessing gambar untuk MNIST"""
    # Baca gambar
    img = cv2.imread(image_path)
    
    # Convert ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize ke 28x28
    resized = cv2.resize(gray, (28, 28))
    
    # Normalize ke [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Reshape ke format [1, 1, 28, 28] - (batch, channel, height, width)
    input_data = normalized.reshape(1, 1, 28, 28)
    
    return input_data

def softmax(x):
    """Apply softmax to convert logits to probabilities"""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

def predict(image_path):
    """Prediksi digit dari gambar"""
    # Get session
    session = _get_session()
    
    # Preprocess
    input_data = preprocess_image(image_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    result = session.run(None, {input_name: input_data})
    
    # Get prediction and apply softmax to normalize probabilities
    output = result[0]
    probabilities = softmax(output[0])  # Apply softmax to get proper probabilities
    predicted_digit = np.argmax(probabilities)
    
    return predicted_digit, probabilities

# Test prediksi
if __name__ == "__main__":
    # Load session and print info
    session = _get_session()
    print("=== Info Model ===")
    print("Input name:", session.get_inputs()[0].name)
    print("Input shape:", session.get_inputs()[0].shape)
    print("Output name:", session.get_outputs()[0].name)
    print("Output shape:", session.get_outputs()[0].shape)
    
    # Ganti dengan path gambar Anda
    image_path = "test_digit.png"
    
    try:
        digit, probs = predict(image_path)
        
        print("\n=== Hasil Prediksi ===")
        print(f"Digit yang diprediksi: {digit}")
        print(f"Confidence: {probs[digit]:.4f}")
        print("\nProbabilitas semua digit:")
        for i, prob in enumerate(probs):
            print(f"  Digit {i}: {prob:.4f}")
    except Exception as e:
        print(f"Error: {e}")