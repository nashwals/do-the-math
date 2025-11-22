import numpy as np
import cv2
import onnxruntime as ort

class DigitRecognizer:
    
    def __init__(self, model_path="models/mnist_model.onnx"):
        """Initialize ONNX digit recognizer"""
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            print(f"✓ ONNX model loaded: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def recognize_multi_digit(self, canvas, max_digits=3):
        """
        Recognize multiple digits from canvas
        
        Args:
            canvas: BGR image (numpy array)
            max_digits: Maximum number of digits to detect
        
        Returns:
            (result_string, average_confidence) or (None, 0.0)
        """
        digit_bboxes = self._find_all_digits(canvas)
        
        if not digit_bboxes:
            return None, 0.0
        
        # Limit to max_digits (prioritize larger contours)
        if len(digit_bboxes) > max_digits:
            digit_bboxes.sort(key=lambda box: box[2] * box[3], reverse=True)
            digit_bboxes = digit_bboxes[:max_digits]
        
        # Sort left to right
        digit_bboxes.sort(key=lambda box: box[0])
        
        digits = []
        confidences = []
        
        for bbox in digit_bboxes:
            img_28x28 = self._preprocess(canvas, bbox)
            digit, confidence = self._predict(img_28x28)
            
            # ✨ Skip low confidence predictions
            if confidence < 0.3:
                continue
            
            digits.append(str(digit))
            confidences.append(confidence)
        
        if not digits:
            return None, 0.0
        
        result_string = "".join(digits)
        avg_confidence = sum(confidences) / len(confidences)
        
        return result_string, avg_confidence
    
    def _find_all_digits(self, canvas):
        """
        Find bounding boxes for all digits in canvas
        
        Returns:
            List of (x, y, w, h) tuples
        """
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        
        # ✨ Apply Gaussian blur untuk reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # ✨ Adaptive thresholding untuk better contrast
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # ✨ Morphological operations untuk clean up
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        bboxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # ✨ Filter criteria
            # 1. Minimum size
            if w < 15 or h < 15:
                continue
            
            # 2. Maximum size (full screen drawings)
            if w > canvas.shape[1] * 0.8 or h > canvas.shape[0] * 0.8:
                continue
            
            # 3. Aspect ratio check
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 2.5 or aspect_ratio < 0.2:
                continue
            
            # 4. Minimum area
            if area < 150:  # ✅ Lowered from 400
                continue
            
            # 5. ✨ NEW: Solidity check (density of contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.25:  # Too sparse/scattered
                continue
            
            # Add padding
            padding = 25  # ✅ Increased from 20
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(canvas.shape[1] - x_padded, w + 2 * padding)
            h_padded = min(canvas.shape[0] - y_padded, h + 2 * padding)
            
            bboxes.append((x_padded, y_padded, w_padded, h_padded))
        
        return bboxes
    
    def _preprocess(self, canvas, bbox):
        """
        Preprocess digit to MNIST format (28x28 grayscale, normalized)
        
        Returns:
            numpy array of shape (1, 1, 28, 28)
        """
        x, y, w, h = bbox
        
        # Extract ROI
        roi = canvas[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # ✨ Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # ✨ Adaptive threshold
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # ✨ Morphological closing untuk connect broken strokes
        kernel = np.ones((2, 2), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Create square image (preserve aspect ratio)
        size = max(gray.shape)
        square = np.zeros((size, size), dtype=np.uint8)
        
        y_offset = (size - gray.shape[0]) // 2
        x_offset = (size - gray.shape[1]) // 2
        square[y_offset:y_offset+gray.shape[0], 
               x_offset:x_offset+gray.shape[1]] = gray
        
        # Resize to 28x28
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # ✨ Invert if background is dark
        # MNIST expects white digit on black background
        if np.mean(resized) > 127:  # Currently black digit on white bg
            resized = 255 - resized  # Invert
        
        # Normalize to 0-1
        normalized = resized.astype('float32') / 255.0
        
        # Reshape for ONNX input (1, 1, 28, 28)
        img_input = normalized.reshape(1, 1, 28, 28)
        
        return img_input
    
    def _predict(self, img_array):
        """
        Run ONNX inference
        
        Returns:
            (predicted_digit, confidence)
        """
        outputs = self.session.run(None, {self.input_name: img_array})
        probs = outputs[0][0]
        
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        
        return digit, confidence
    
    def visualize_preprocessing(self, canvas, bbox, window_name="Debug"):
        """
        Debug function to visualize preprocessing steps
        
        Args:
            canvas: Original canvas
            bbox: Bounding box (x, y, w, h)
            window_name: OpenCV window name
        """
        x, y, w, h = bbox
        roi = canvas[y:y+h, x:x+w]
        
        # Step 1: Original ROI
        cv2.imshow(f"{window_name} - 1. Original", roi)
        
        # Step 2: Grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cv2.imshow(f"{window_name} - 2. Grayscale", gray)
        
        # Step 3: After blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imshow(f"{window_name} - 3. Blurred", blurred)
        
        # Step 4: After threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        cv2.imshow(f"{window_name} - 4. Thresholded", thresh)
        
        # Step 5: Final 28x28
        img_28x28 = self._preprocess(canvas, bbox)
        final = (img_28x28[0, 0] * 255).astype(np.uint8)
        final_scaled = cv2.resize(final, (280, 280), 
                                  interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"{window_name} - 5. Final 28x28", final_scaled)
        
        print(f"Preprocessing complete. Press any key in debug windows...")