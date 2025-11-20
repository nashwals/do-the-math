import numpy as np
import cv2
import onnxruntime as ort

class DigitRecognizer:
    
    def __init__(self, model_path="models/mnist_model.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def recognize(self, canvas):
        # Recognize single digit dari canvas
        # Return: (digit, confidence) atau (None, 0.0) jika gagal
        
        bbox = self._find_digit(canvas)
        if bbox is None:
            return None, 0.0
        
        img_28x28 = self._preprocess(canvas, bbox)
        digit, confidence = self._predict(img_28x28)
        
        return digit, confidence
    
    def recognize_multi_digit(self, canvas, max_digits=3):
        # Recognize multiple digits dari canvas
        # Return: (number_string, average_confidence) atau (None, 0.0)
        
        digit_bboxes = self._find_all_digits(canvas)
        
        if not digit_bboxes:
            return None, 0.0
        
        if len(digit_bboxes) > max_digits:
            digit_bboxes.sort(key=lambda box: box[2] * box[3], reverse=True)
            digit_bboxes = digit_bboxes[:max_digits]
        
        # Sort dari kiri ke kanan
        digit_bboxes.sort(key=lambda box: box[0])
        
        digits = []
        confidences = []
        
        for bbox in digit_bboxes:
            img_28x28 = self._preprocess(canvas, bbox)
            digit, confidence = self._predict(img_28x28)
            
            digits.append(str(digit))
            confidences.append(confidence)
        
        result_string = "".join(digits)
        avg_confidence = sum(confidences) / len(confidences)
        
        return result_string, avg_confidence
    
    def _find_digit(self, canvas):
        # Cari bounding box dari single digit terbesar
        
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        if w < 30 or h < 30:
            return None
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(canvas.shape[1] - x, w + 2 * padding)
        h = min(canvas.shape[0] - y, h + 2 * padding)
        
        return (x, y, w, h)
    
    def _find_all_digits(self, canvas):
        # Find bounding boxes untuk semua digit
        
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        bboxes = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter terlalu kecil
            if w < 20 or h < 20:
                continue
            
            # Filter terlalu besar
            if w > 500 or h > 500:
                continue
            
            # Filter aspect ratio tidak wajar
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 3.0 or aspect_ratio < 0.1:
                continue
            
            # Filter area terlalu kecil
            area = cv2.contourArea(contour)
            if area < 400:
                continue
            
            # Add padding
            padding = 20
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(canvas.shape[1] - x_padded, w + 2 * padding)
            h_padded = min(canvas.shape[0] - y_padded, h + 2 * padding)
            
            bboxes.append((x_padded, y_padded, w_padded, h_padded))
        
        return bboxes
    
    def _preprocess(self, canvas, bbox):
        # Preprocess digit ke format MNIST (28x28 grayscale, normalized)
        
        x, y, w, h = bbox
        
        roi = canvas[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Buat square image untuk preserve aspect ratio
        size = max(gray.shape)
        square = np.zeros((size, size), dtype=np.uint8)
        
        y_offset = (size - gray.shape[0]) // 2
        x_offset = (size - gray.shape[1]) // 2
        square[y_offset:y_offset+gray.shape[0], 
               x_offset:x_offset+gray.shape[1]] = gray
        
        # Resize to 28x28
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize to 0-1
        normalized = resized.astype('float32') / 255.0
        
        # Reshape untuk ONNX input (1, 1, 28, 28)
        img_input = normalized.reshape(1, 1, 28, 28)
        
        return img_input
    
    def _predict(self, img_array):
        # Run ONNX inference
        
        outputs = self.session.run(None, {self.input_name: img_array})
        probs = outputs[0][0]
        
        digit = int(np.argmax(probs))
        confidence = float(probs[digit])
        
        return digit, confidence