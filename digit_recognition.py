"""
Modul Digit Recognition untuk Mengenali Angka Tulisan Tangan
Menggunakan ONNX Runtime untuk inference model MNIST
"""

import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
import os


def download_onnx_model():
    """
    Download pre-trained MNIST model dalam format ONNX dari repository.
    
    Returns:
        str: Path ke file model yang sudah didownload
        
    Raises:
        Exception: Jika download gagal
    """
    model_dir = "models"
    model_path = os.path.join(model_dir, "mnist-8.onnx")
    
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(model_path):
        print(f"Model ONNX sudah ada di {model_path}")
        return model_path
    
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"
    
    try:
        print("Downloading MNIST ONNX model...")
        print(f"URL: {model_url}")
        print("Ini mungkin memakan waktu beberapa detik...")
        
        urllib.request.urlretrieve(model_url, model_path)
        
        print(f"Model berhasil didownload ke {model_path}")
        return model_path
    
    except Exception as e:
        print(f"Error saat mendownload model: {str(e)}")
        raise


def load_model():
    """
    Memuat model MNIST ONNX yang sudah dilatih atau mendownload jika belum ada.
    
    Returns:
        onnxruntime.InferenceSession: Model MNIST yang siap digunakan untuk prediksi
        
    Raises:
        Exception: Jika terjadi error saat loading model
    """
    try:
        model_path = download_onnx_model()
        
        print("\nLoading ONNX model...")
        session = ort.InferenceSession(model_path)
        
        print("Model berhasil dimuat!")
        print(f"Input name: {session.get_inputs()[0].name}")
        print(f"Input shape: {session.get_inputs()[0].shape}")
        print(f"Output name: {session.get_outputs()[0].name}")
        
        return session
    
    except Exception as e:
        print(f"Error saat loading model: {str(e)}")
        raise


def find_digit_bboxes(canvas, min_area=200, max_digits=3):
    """
    Mencari bounding boxes untuk semua digit di canvas untuk multi-digit recognition.
    Optimized untuk mendeteksi digit terpisah dengan lebih baik.
    
    Args:
        canvas: Canvas berisi gambar digit dalam format BGR
        min_area: Minimum area contour yang dianggap valid (dinaikkan untuk filter noise)
        max_digits: Maximum jumlah digit yang akan dideteksi
        
    Returns:
        list: List of bounding boxes (x, y, w, h) sorted dari kiri ke kanan
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    _, thresh = cv2.threshold(denoised, 30, 255, cv2.THRESH_BINARY)
    
    # Reduced morphological operations untuk preserve digit separation
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
        
        # Stricter size constraints untuk filter noise
        if w < 20 or h < 20:
            continue
        if w > canvas.shape[1] * 0.9 or h > canvas.shape[0] * 0.9:
            continue
        
        # Stricter aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 2.5 or aspect_ratio < 0.2:
            continue
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if solidity < 0.3:
            continue
        
        # Increased padding for better digit capture
        padding = 30
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(canvas.shape[1] - x_padded, w + 2 * padding)
        h_padded = min(canvas.shape[0] - y_padded, h + 2 * padding)
        
        bboxes.append((x_padded, y_padded, w_padded, h_padded))
    
    if len(bboxes) > max_digits:
        bboxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        bboxes = bboxes[:max_digits]
    
    bboxes.sort(key=lambda box: box[0])
    
    return bboxes


def preprocess_single_digit(canvas, bbox):
    """
    Memproses single digit dari bounding box untuk input ke model MNIST.
    
    Args:
        canvas: Canvas berisi gambar digit dalam format BGR
        bbox: Bounding box (x, y, w, h)
        
    Returns:
        numpy.ndarray: Gambar yang sudah diproses dengan shape (1, 1, 28, 28)
    """
    x, y, w, h = bbox
    
    roi = canvas[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    _, thresh = cv2.threshold(denoised, 30, 255, cv2.THRESH_BINARY)
    
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel_dilate, iterations=2)
    
    kernel_erode = np.ones((2, 2), np.uint8)
    processed = cv2.erode(dilated, kernel_erode, iterations=1)
    
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
    
    resized_28 = cv2.resize(square_canvas, (28, 28), interpolation=cv2.INTER_AREA)
    
    normalized = resized_28.astype("float32") / 255.0
    
    processed_img = normalized.reshape(1, 1, 28, 28)
    
    return processed_img


def recognize_multi_digit(session, canvas, max_digits=3):
    """
    Mengenali multiple digits dari canvas untuk angka lebih dari 1 digit.
    
    Args:
        session: ONNX model session
        canvas: Canvas berisi gambar digit dalam format BGR
        max_digits: Maximum jumlah digit yang akan dideteksi
        
    Returns:
        tuple: (result_string, average_confidence)
            result_string: String angka hasil prediksi contoh "123"
            average_confidence: Rata-rata confidence dalam persen
            None, 0.0 jika tidak ada digit terdeteksi
    """
    bboxes = find_digit_bboxes(canvas, min_area=200, max_digits=max_digits)
    
    if not bboxes:
        return None, 0.0
    
    digits = []
    confidences = []
    
    input_name = session.get_inputs()[0].name
    
    for bbox in bboxes:
        processed = preprocess_single_digit(canvas, bbox)
        
        outputs = session.run(None, {input_name: processed})
        predictions = outputs[0][0]
        
        exp_pred = np.exp(predictions - np.max(predictions))
        probabilities = exp_pred / np.sum(exp_pred)
        
        digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100
        
        # Raised confidence threshold untuk avoid false positives
        if confidence < 40:
            continue
        
        digits.append(str(digit))
        confidences.append(confidence)
    
    if not digits:
        return None, 0.0
    
    result_string = "".join(digits)
    avg_confidence = sum(confidences) / len(confidences)
    
    return result_string, avg_confidence


def preprocess_canvas(canvas, debug_mode=False):
    """
    Memproses canvas gambar untuk input ke model MNIST.
    Fungsi ini untuk single digit recognition.
    Untuk multi-digit gunakan recognize_multi_digit().
    
    Args:
        canvas: Canvas berisi gambar digit dalam format BGR
        debug_mode: Jika True, tampilkan visualisasi preprocessing steps
        
    Returns:
        numpy.ndarray: Gambar yang sudah diproses dengan shape (1, 1, 28, 28)
        None: Jika tidak ada gambar yang terdeteksi di canvas
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    _, thresh = cv2.threshold(denoised, 30, 255, cv2.THRESH_BINARY)
    
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel_dilate, iterations=2)
    
    kernel_erode = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(dilated, kernel_erode, iterations=1)
    
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("Tidak ada gambar yang terdeteksi di canvas")
        return None
    
    min_contour_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    if len(valid_contours) == 0:
        print("Gambar terlalu kecil atau hanya noise")
        return None
    
    all_contours = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    
    padding_percent = 0.2
    padding_w = int(w * padding_percent)
    padding_h = int(h * padding_percent)
    
    x = max(0, x - padding_w)
    y = max(0, y - padding_h)
    w = min(canvas.shape[1] - x, w + 2 * padding_w)
    h = min(canvas.shape[0] - y, h + 2 * padding_h)
    
    cropped = eroded[y:y+h, x:x+w]
    
    target_size = 56
    max_side = max(w, h)
    
    scale_factor = (target_size * 0.8) / max_side
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    resized_cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    square_canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    square_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_cropped
    
    resized = cv2.resize(square_canvas, (28, 28), interpolation=cv2.INTER_AREA)
    
    normalized = resized.astype("float32") / 255.0
    
    processed = normalized.reshape(1, 1, 28, 28)
    
    if debug_mode:
        visualize_preprocessing(gray, denoised, thresh, dilated, eroded, 
                               cropped, square_canvas, resized)
    
    return processed


def predict_digit(session, processed_image):
    """
    Memprediksi digit dari gambar yang telah diproses menggunakan ONNX model.
    
    Args:
        session: ONNX model session
        processed_image: Gambar yang sudah diproses dengan shape (1, 1, 28, 28)
        
    Returns:
        tuple: (predicted_digit, confidence)
            predicted_digit: Angka hasil prediksi (0-9)
            confidence: Tingkat keyakinan prediksi dalam persen (0-100)
    """
    input_name = session.get_inputs()[0].name
    
    outputs = session.run(None, {input_name: processed_image})
    predictions = outputs[0][0]
    
    exp_predictions = np.exp(predictions - np.max(predictions))
    probabilities = exp_predictions / np.sum(exp_predictions)
    
    predicted_digit = np.argmax(probabilities)
    confidence = np.max(probabilities) * 100
    
    return int(predicted_digit), float(confidence)


def display_prediction_result(predicted_digit, confidence):
    """
    Menampilkan hasil prediksi ke terminal dengan format yang rapi.
    
    Args:
        predicted_digit: Angka hasil prediksi
        confidence: Tingkat keyakinan prediksi dalam persen (0-100)
    """
    print("\nHASIL PREDIKSI DIGIT")
    print(f"Angka terdeteksi: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    print()


def visualize_preprocessing(gray, denoised, thresh, dilated, eroded, 
                            cropped, square_canvas, resized):
    """
    Menampilkan visualisasi dari setiap step preprocessing untuk debugging.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.suptitle('Preprocessing Pipeline Visualization', fontsize=16)
        
        steps = [
            (gray, 'Step 1: Grayscale'),
            (denoised, 'Step 2: Denoised'),
            (thresh, 'Step 3: Threshold'),
            (dilated, 'Step 4: Dilated'),
            (eroded, 'Step 5: Eroded'),
            (cropped, 'Step 6: Cropped'),
            (square_canvas, 'Step 7: Square Canvas'),
            (resized, 'Step 8: Final 28x28')
        ]
        
        for idx, (img, title) in enumerate(steps):
            row = idx // 4
            col = idx % 4
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nVisualisasi preprocessing ditampilkan.")
    except ImportError:
        print("Matplotlib tidak tersedia. Skip visualisasi.")