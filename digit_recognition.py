"""
Modul Digit Recognition untuk Mengenali Angka Tulisan Tangan
Menggunakan ONNX Runtime untuk inference model MNIST
Version: 2.0 - Cleaned & Optimized
"""

import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
import os
from datetime import datetime


def download_onnx_model():
    """Download pre-trained MNIST model dalam format ONNX dari repository."""
    model_dir = "models"
    model_path = os.path.join(model_dir, "mnist-8.onnx")
    
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"✅ Model ONNX sudah ada: {model_path}")
        return model_path
    
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"
    
    try:
        print("📥 Downloading MNIST ONNX model...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✅ Model downloaded: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        raise


def load_model():
    """Memuat model MNIST ONNX."""
    try:
        model_path = download_onnx_model()
        session = ort.InferenceSession(model_path)
        print("✅ Model loaded successfully!")
        return session
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise


def find_digit_bboxes(canvas, min_area=200, max_digits=3):
    """
    Mencari bounding boxes untuk semua digit di canvas.
    
    Returns:
        list: List of bounding boxes (x, y, w, h) sorted kiri ke kanan
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    _, thresh = cv2.threshold(denoised, 30, 255, cv2.THRESH_BINARY)
    
    # Minimal morphological ops untuk preserve digit separation
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Size constraints
        if w < 20 or h < 20:
            continue
        if w > canvas.shape[1] * 0.9 or h > canvas.shape[0] * 0.9:
            continue
        
        # Aspect ratio check
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 2.5 or aspect_ratio < 0.2:
            continue
        
        # Solidity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if solidity < 0.3:
            continue
        
        # Add padding
        padding = 30
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(canvas.shape[1] - x_padded, w + 2 * padding)
        h_padded = min(canvas.shape[0] - y_padded, h + 2 * padding)
        
        bboxes.append((x_padded, y_padded, w_padded, h_padded))
    
    # Keep only top N largest if too many
    if len(bboxes) > max_digits:
        bboxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        bboxes = bboxes[:max_digits]
    
    # Sort left to right
    bboxes.sort(key=lambda box: box[0])
    
    return bboxes


def preprocess_single_digit(canvas, bbox, save_visualization=False, output_dir="debug_output", digit_index=0):
    """
    Memproses single digit dari bounding box untuk input ke model MNIST.
    
    Args:
        canvas: Canvas berisi gambar digit (BGR)
        bbox: Bounding box (x, y, w, h)
        save_visualization: If True, save preprocessing steps as image
        output_dir: Directory untuk save visualisasi
        digit_index: Index digit (untuk multi-digit)
        
    Returns:
        numpy.ndarray: Gambar processed (1, 1, 28, 28)
        dict: Visualization images (if save_visualization=True)
    """
    x, y, w, h = bbox
    
    # Step 1: Extract ROI
    roi = canvas[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Step 3: Threshold
    _, thresh = cv2.threshold(denoised, 30, 255, cv2.THRESH_BINARY)
    
    # Step 4: Dilate
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel_dilate, iterations=2)
    
    # Step 5: Erode
    kernel_erode = np.ones((2, 2), np.uint8)
    processed = cv2.erode(dilated, kernel_erode, iterations=1)
    
    # Step 6: Resize to square canvas
    target_size = 56
    max_side = max(w, h)
    scale_factor = (target_size * 0.8) / max_side
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    resized = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    square_canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    square_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Step 7: Final resize to 28x28
    resized_28 = cv2.resize(square_canvas, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Step 8: Normalize
    normalized = resized_28.astype("float32") / 255.0
    processed_img = normalized.reshape(1, 1, 28, 28)
    
    # Save visualization if requested
    if save_visualization:
        save_preprocessing_visualization(
            gray, denoised, thresh, dilated, processed, 
            square_canvas, resized_28,
            output_dir=output_dir,
            digit_index=digit_index
        )
    
    return processed_img


def save_preprocessing_visualization(gray, denoised, thresh, dilated, eroded, 
                                     square_canvas, final_28x28,
                                     output_dir="debug_output", digit_index=0):
    """
    Save preprocessing steps sebagai gambar grid.
    Setiap submit jawaban, akan create file baru dengan timestamp.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp untuk filename unik
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"preprocessing_digit{digit_index}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Create visualization grid (2 rows x 4 cols)
    steps = [
        (gray, '1. Grayscale'),
        (denoised, '2. Denoised'),
        (thresh, '3. Threshold'),
        (dilated, '4. Dilated'),
        (eroded, '5. Eroded'),
        (square_canvas, '6. Square 56x56'),
        (final_28x28, '7. Final 28x28'),
        (final_28x28, '8. To Model')  # Duplicate for symmetry
    ]
    
    # Size untuk setiap cell
    cell_size = 200
    grid_w = 4 * cell_size
    grid_h = 2 * cell_size
    
    # Create white canvas
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    for idx, (img, title) in enumerate(steps):
        row = idx // 4
        col = idx % 4
        
        # Calculate position
        x_start = col * cell_size
        y_start = row * cell_size
        
        # Resize image to fit cell (leave space for title)
        img_h = cell_size - 40
        img_w = cell_size - 20
        
        # Convert to BGR if grayscale
        if len(img.shape) == 2:
            img_resized = cv2.resize(img, (img_w, img_h))
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.resize(img, (img_w, img_h))
        
        # Place image in grid
        grid[y_start+30:y_start+30+img_h, x_start+10:x_start+10+img_w] = img_bgr
        
        # Add title
        cv2.putText(grid, title, (x_start + 10, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add border around cell
        cv2.rectangle(grid, (x_start, y_start), 
                     (x_start + cell_size, y_start + cell_size),
                     (200, 200, 200), 1)
    
    # Save grid
    cv2.imwrite(filepath, grid)
    print(f"📸 Preprocessing visualization saved: {filepath}")
    
    return filepath


def recognize_multi_digit(session, canvas, max_digits=2, save_debug=False):
    """
    Mengenali multiple digits dari canvas.
    
    Args:
        session: ONNX model session
        canvas: Canvas berisi gambar digit (BGR)
        max_digits: Maximum jumlah digit yang akan dideteksi
        save_debug: If True, save preprocessing visualization
        
    Returns:
        tuple: (result_string, average_confidence)
            - result_string: String angka hasil prediksi (e.g., "42")
            - average_confidence: Rata-rata confidence dalam persen
            - (None, 0.0) jika tidak ada digit terdeteksi
    """
    bboxes = find_digit_bboxes(canvas, min_area=200, max_digits=max_digits)
    
    if not bboxes:
        print("⚠️ No digits detected")
        return None, 0.0
    
    digits = []
    confidences = []
    
    input_name = session.get_inputs()[0].name
    
    for i, bbox in enumerate(bboxes):
        # Process digit with optional visualization
        processed = preprocess_single_digit(
            canvas, bbox, 
            save_visualization=save_debug,
            digit_index=i
        )
        
        # Run inference
        outputs = session.run(None, {input_name: processed})
        predictions = outputs[0][0]
        
        # Softmax
        exp_pred = np.exp(predictions - np.max(predictions))
        probabilities = exp_pred / np.sum(exp_pred)
        
        digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100
        
        # Confidence threshold
        if confidence < 40:
            print(f"⚠️ Digit {i}: Low confidence ({confidence:.1f}%) - skipped")
            continue
        
        digits.append(str(digit))
        confidences.append(confidence)
        print(f"✅ Digit {i}: {digit} ({confidence:.1f}%)")
    
    if not digits:
        return None, 0.0
    
    result_string = "".join(digits)
    avg_confidence = sum(confidences) / len(confidences)
    
    print(f"🎯 Final result: {result_string} (avg confidence: {avg_confidence:.1f}%)")
    
    return result_string, avg_confidence