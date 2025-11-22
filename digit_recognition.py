"""
Modul Digit Recognition untuk Mengenali Angka Tulisan Tangan (ONNX Version)
============================================================================

Modul ini menyediakan fungsi-fungsi untuk:
1. Memuat model MNIST pre-trained dalam format ONNX
2. Preprocessing canvas gambar untuk input model
3. Prediksi digit dari gambar yang telah diproses

Menggunakan ONNX Runtime untuk inference yang lebih cepat dan portable.
"""

import cv2
import numpy as np
import onnxruntime as ort
import urllib.request
import os


def download_onnx_model():
    """
    Download pre-trained MNIST model dalam format ONNX dari repository.
    
    Model akan didownload dari ONNX Model Zoo dan disimpan di folder models/.
    Model ini sudah dilatih dengan dataset MNIST dan memiliki akurasi ~99%.
    
    Returns:
        str: Path ke file model yang sudah didownload
        
    Raises:
        Exception: Jika download gagal
    """
    model_dir = "models"
    model_path = os.path.join(model_dir, "mnist-8.onnx")
    
    # Buat folder models jika belum ada
    os.makedirs(model_dir, exist_ok=True)
    
    # Jika model sudah ada, tidak perlu download lagi
    if os.path.exists(model_path):
        print(f"✓ Model ONNX sudah ada di {model_path}")
        return model_path
    
    # URL model MNIST dari ONNX Model Zoo
    model_url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"
    
    try:
        print("Downloading MNIST ONNX model...")
        print(f"URL: {model_url}")
        print("Ini mungkin memakan waktu beberapa detik...")
        
        # Download model
        urllib.request.urlretrieve(model_url, model_path)
        
        print(f"✓ Model berhasil didownload ke {model_path}")
        return model_path
    
    except Exception as e:
        print(f"✗ Error saat mendownload model: {str(e)}")
        raise


def load_model():
    """
    Memuat model MNIST ONNX yang sudah dilatih atau mendownload jika belum ada.
    
    Fungsi ini akan:
    1. Cek apakah file model sudah ada
    2. Jika belum ada, download dari ONNX Model Zoo
    3. Load model menggunakan ONNX Runtime
    
    Returns:
        onnxruntime.InferenceSession: Model MNIST yang siap digunakan untuk prediksi
        
    Raises:
        Exception: Jika terjadi error saat loading model
    """
    try:
        # Download model jika belum ada
        model_path = download_onnx_model()
        
        # Load model dengan ONNX Runtime
        print("\nLoading ONNX model...")
        session = ort.InferenceSession(model_path)
        
        # Tampilkan informasi model
        print("✓ Model berhasil dimuat!")
        print(f"Input name: {session.get_inputs()[0].name}")
        print(f"Input shape: {session.get_inputs()[0].shape}")
        print(f"Output name: {session.get_outputs()[0].name}")
        
        return session
    
    except Exception as e:
        print(f"✗ Error saat loading model: {str(e)}")
        raise


def preprocess_canvas(canvas, debug_mode=False):
    """
    Memproses canvas gambar untuk input ke model MNIST dengan teknik yang robust.
    
    Tahapan preprocessing (OPTIMIZED):
    1. Konversi ke grayscale
    2. Denoising untuk mengurangi noise
    3. Thresholding untuk isolasi digit
    4. Morphological operations untuk memperkuat stroke
    5. Deteksi dan crop bounding box
    6. Centering di square canvas dengan padding optimal
    7. Resize ke 28x28 dengan anti-aliasing
    8. Normalisasi sesuai format ONNX MNIST model
    9. Reshape untuk input model ONNX
    
    Args:
        canvas (numpy.ndarray): Canvas berisi gambar digit (BGR format)
        debug_mode (bool): Jika True, tampilkan visualisasi preprocessing steps
        
    Returns:
        numpy.ndarray: Gambar yang sudah diproses dengan shape (1, 1, 28, 28) untuk ONNX
        None: Jika tidak ada gambar yang terdeteksi di canvas
        
    Example:
        >>> processed_img = preprocess_canvas(canvas, debug_mode=True)
        >>> if processed_img is not None:
        >>>     prediction = session.run(None, {input_name: processed_img})
    """
    # Step 1: Konversi ke grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Denoising - mengurangi noise pada gambar
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Step 3: Threshold
    _, thresh = cv2.threshold(denoised, 30, 255, cv2.THRESH_BINARY)
    
    # Step 4: Morphological operations untuk memperkuat stroke
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel_dilate, iterations=2)
    
    kernel_erode = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(dilated, kernel_erode, iterations=1)
    
    # Step 5: Cari contours dari area yang digambar
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Jika tidak ada gambar yang terdeteksi
    if len(contours) == 0:
        print("⚠ Tidak ada gambar yang terdeteksi di canvas")
        return None
    
    # Step 6: Filter contours yang terlalu kecil (noise)
    min_contour_area = 100
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    
    if len(valid_contours) == 0:
        print("⚠ Gambar terlalu kecil atau hanya noise")
        return None
    
    # Gabungkan semua valid contours untuk mendapatkan bounding box
    all_contours = np.vstack(valid_contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    
    # Step 7: Tambahkan padding proporsional (20% dari dimensi)
    padding_percent = 0.2
    padding_w = int(w * padding_percent)
    padding_h = int(h * padding_percent)
    
    x = max(0, x - padding_w)
    y = max(0, y - padding_h)
    w = min(canvas.shape[1] - x, w + 2 * padding_w)
    h = min(canvas.shape[0] - y, h + 2 * padding_h)
    
    # Crop gambar sesuai bounding box
    cropped = eroded[y:y+h, x:x+w]
    
    # Step 8: Buat square canvas dengan ukuran lebih besar untuk quality
    target_size = 56  # 2x ukuran final untuk anti-aliasing
    max_side = max(w, h)
    
    # Hitung scale factor agar digit memenuhi ~80% area
    scale_factor = (target_size * 0.8) / max_side
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # Resize cropped image
    resized_cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Buat square canvas dan center gambar
    square_canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    square_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_cropped
    
    # Step 9: Resize ke 28x28 dengan anti-aliasing
    resized = cv2.resize(square_canvas, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Step 10: Normalisasi untuk ONNX model
    # ONNX MNIST model expects input range [0, 1] with float32
    normalized = resized.astype("float32") / 255.0
    
    # Step 11: Reshape untuk ONNX model: (1, 1, 28, 28)
    # Format: (batch_size, channels, height, width)
    processed = normalized.reshape(1, 1, 28, 28)
    
    # Debug mode: Tampilkan visualisasi preprocessing steps
    if debug_mode:
        visualize_preprocessing(gray, denoised, thresh, dilated, eroded, 
                               cropped, square_canvas, resized)
    
    return processed


def predict_digit(session, processed_image):
    """
    Memprediksi digit dari gambar yang telah diproses menggunakan ONNX model.
    
    Fungsi ini menggunakan ONNX Runtime untuk inference dan
    mengembalikan hasil prediksi beserta confidence score.
    
    Args:
        session (onnxruntime.InferenceSession): ONNX model session
        processed_image (numpy.ndarray): Gambar yang sudah diproses dengan shape (1, 1, 28, 28)
        
    Returns:
        tuple: (predicted_digit, confidence)
            - predicted_digit (int): Angka hasil prediksi (0-9)
            - confidence (float): Tingkat keyakinan prediksi dalam persen (0-100)
            
    Example:
        >>> digit, conf = predict_digit(session, processed_img)
        >>> print(f"Prediksi: {digit} (Confidence: {conf:.2f}%)")
    """
    # Get input name dari model
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: processed_image})
    
    # Output dari ONNX MNIST model adalah logits atau probabilities
    # Shape: (1, 10) untuk 10 kelas digit (0-9)
    predictions = outputs[0][0]
    
    # Apply softmax jika output adalah logits (untuk mendapatkan probabilities)
    # Jika sudah probabilities, softmax tidak akan mengubah urutan
    exp_predictions = np.exp(predictions - np.max(predictions))  # Subtract max for numerical stability
    probabilities = exp_predictions / np.sum(exp_predictions)
    
    # Ambil indeks dengan probabilitas tertinggi
    predicted_digit = np.argmax(probabilities)
    
    # Ambil nilai confidence (probabilitas tertinggi dalam persen)
    confidence = np.max(probabilities) * 100
    
    return int(predicted_digit), float(confidence)


def display_prediction_result(predicted_digit, confidence):
    """
    Menampilkan hasil prediksi ke terminal dengan format yang rapi.
    
    Args:
        predicted_digit (int): Angka hasil prediksi (0-9)
        confidence (float): Tingkat keyakinan prediksi dalam persen (0-100)
    """
    print("\n" + "=" * 50)
    print("HASIL PREDIKSI DIGIT")
    print("=" * 50)
    print(f"Angka terdeteksi: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 50 + "\n")


def visualize_preprocessing(gray, denoised, thresh, dilated, eroded, 
                            cropped, square_canvas, resized):
    """
    Menampilkan visualisasi dari setiap step preprocessing untuk debugging.
    
    Fungsi ini berguna untuk:
    - Debugging preprocessing pipeline
    - Memahami transformasi yang terjadi pada gambar
    - Mengidentifikasi masalah pada preprocessing
    
    Args:
        gray: Gambar grayscale
        denoised: Gambar setelah denoising
        thresh: Gambar setelah thresholding
        dilated: Gambar setelah dilation
        eroded: Gambar setelah erosion
        cropped: Gambar yang sudah di-crop
        square_canvas: Gambar di square canvas
        resized: Gambar yang sudah di-resize ke 28x28
    """
    import matplotlib.pyplot as plt
    
    # Buat figure dengan subplot untuk setiap step
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle('Preprocessing Pipeline Visualization (ONNX)', fontsize=16)
    
    # Plot setiap step
    steps = [
        (gray, 'Step 1: Grayscale'),
        (denoised, 'Step 2: Denoised'),
        (thresh, 'Step 3: Threshold'),
        (dilated, 'Step 4: Dilated'),
        (eroded, 'Step 5: Eroded'),
        (cropped, 'Step 6: Cropped'),
        (square_canvas, 'Step 7: Square Canvas'),
        (resized, 'Step 8: Final (28x28)')
    ]
    
    for idx, (img, title) in enumerate(steps):
        row = idx // 4
        col = idx % 4
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n[DEBUG] Visualisasi preprocessing ditampilkan.")
    print("[DEBUG] Tutup window visualisasi untuk melanjutkan.")


# Test function untuk memastikan modul berjalan dengan baik
if __name__ == "__main__":
    print("Testing Digit Recognition Module (ONNX Version)...")
    print("-" * 60)
    
    # Test loading model
    try:
        session = load_model()
        print("\n✓ Model loading test: PASSED")
        print(f"✓ Model ready for inference")
    except Exception as e:
        print(f"\n✗ Model loading test: FAILED - {str(e)}")
    
    print("\nModule siap digunakan!")
    print("\nTips untuk akurasi lebih baik:")
    print("- Gambar angka dengan jelas dan tidak terlalu miring")
    print("- Pastikan stroke tidak terlalu tipis atau terlalu tebal")
    print("- Gambar di area tengah canvas")
    print("- Gunakan debug_mode=True untuk melihat preprocessing steps")