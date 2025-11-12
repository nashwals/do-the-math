import cv2
import numpy as np
from typing import List, Tuple, Optional


class AirDrawer:
    """
    Class untuk menggambar di canvas virtual menggunakan fingertip tracking.
    
    Attributes:
        canvas: Numpy array untuk menyimpan drawing
        drawing_points: List of points yang digambar
        is_drawing: Flag untuk status drawing
        brush_color: Warna brush (BGR format)
        brush_thickness: Ketebalan brush
    """
    
    def __init__(self, canvas_size: Tuple[int, int] = (720, 1280), 
                 brush_color: Tuple[int, int, int] = (0, 0, 255),
                 brush_thickness: int = 8):
        """
        Initialize Air Drawer.
        
        Args:
            canvas_size: (height, width) ukuran canvas
            brush_color: Warna brush dalam BGR (default: merah)
            brush_thickness: Ketebalan garis brush (default: 8 px)
        """
        self.height, self.width = canvas_size
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self.drawing_points = []
        self.all_strokes = []  # List of lists untuk menyimpan semua stroke
        self.is_drawing = False
        
        self.brush_color = brush_color
        self.brush_thickness = brush_thickness
        
        print(f"[AirDrawer] Initialized with size={canvas_size}, "
              f"color={brush_color}, thickness={brush_thickness}")
    
    def start_drawing(self, point: Tuple[int, int]) -> None:
        """
        Mulai drawing stroke baru.
        
        Args:
            point: (x, y) koordinat starting point
        """
        self.is_drawing = True
        self.drawing_points = [point]
    
    def add_point(self, point: Tuple[int, int]) -> None:
        """
        Tambahkan point ke current stroke.
        
        Args:
            point: (x, y) koordinat point
        """
        if self.is_drawing:
            self.drawing_points.append(point)
            self._draw_line_segment(point)
    
    def end_drawing(self) -> None:
        """
        Akhiri current stroke dan simpan ke history.
        """
        if self.is_drawing and len(self.drawing_points) > 0:
            self.all_strokes.append(self.drawing_points.copy())
            self.is_drawing = False
    
    def _draw_line_segment(self, current_point: Tuple[int, int]) -> None:
        """
        Gambar garis dari point sebelumnya ke point saat ini.
        
        Args:
            current_point: (x, y) koordinat point saat ini
        """
        if len(self.drawing_points) < 2:
            return
        
        prev_point = self.drawing_points[-2]
        
        cv2.line(self.canvas, prev_point, current_point, 
                self.brush_color, self.brush_thickness, cv2.LINE_AA)
    
    def draw_all_strokes(self) -> None:
        """
        Redraw semua strokes yang tersimpan ke canvas.
        Digunakan untuk refresh canvas.
        """
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for stroke in self.all_strokes:
            for i in range(1, len(stroke)):
                cv2.line(self.canvas, stroke[i-1], stroke[i], 
                        self.brush_color, self.brush_thickness, cv2.LINE_AA)
    
    def get_canvas(self) -> np.ndarray:
        """
        Dapatkan canvas saat ini.
        
        Returns:
            Canvas sebagai numpy array
        """
        return self.canvas
    
    def clear_canvas(self) -> None:
        """
        Hapus semua drawing dari canvas.
        """
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_points = []
        self.all_strokes = []
        self.is_drawing = False
        print("[AirDrawer] Canvas cleared")
    
    def get_digit_roi(self) -> Optional[np.ndarray]:
        """
        Extract region of interest (ROI) yang berisi digit untuk recognition.
        
        Returns:
            ROI sebagai grayscale image atau None jika tidak ada drawing
        """
        if len(self.all_strokes) == 0:
            return None
        
        # Flatten all points
        all_points = []
        for stroke in self.all_strokes:
            all_points.extend(stroke)
        
        if len(all_points) == 0:
            return None
        
        # Find bounding box
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(self.width, max_x + padding)
        max_y = min(self.height, max_y + padding)
        
        # Crop ROI
        roi = self.canvas[min_y:max_y, min_x:max_x]
        
        if roi.size == 0:
            return None
        
        # Convert to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        return roi_gray
    
    def get_preprocessed_digit(self, target_size: Tuple[int, int] = (28, 28)) -> Optional[np.ndarray]:
        """
        Dapatkan digit yang sudah dipreprocess untuk MNIST model.
        
        Args:
            target_size: (height, width) target size (default: 28x28 untuk MNIST)
            
        Returns:
            Preprocessed image sebagai numpy array atau None
        """
        roi = self.get_digit_roi()
        
        if roi is None:
            return None
        
        # Resize dengan aspect ratio preserved
        h, w = roi.shape
        
        if h > w:
            new_h = target_size[0]
            new_w = int(w * (new_h / h))
        else:
            new_w = target_size[1]
            new_h = int(h * (new_w / w))
        
        resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create square canvas with padding
        result = np.zeros(target_size, dtype=np.uint8)
        
        # Center the digit
        y_offset = (target_size[0] - new_h) // 2
        x_offset = (target_size[1] - new_w) // 2
        
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Normalize to 0-1 range
        result = result.astype(np.float32) / 255.0
        
        return result
    
    def overlay_on_frame(self, frame: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Overlay canvas ke frame dengan transparansi.
        
        Args:
            frame: Frame dari webcam
            alpha: Transparansi canvas (0.0 - 1.0)
            
        Returns:
            Frame dengan canvas overlay
        """
        # Ensure same size
        if frame.shape[:2] != (self.height, self.width):
            canvas_resized = cv2.resize(self.canvas, (frame.shape[1], frame.shape[0]))
        else:
            canvas_resized = self.canvas.copy()
        
        # Create mask for non-black pixels in canvas
        gray_canvas = cv2.cvtColor(canvas_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        # Keep frame background where there's no drawing
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        
        # Blend canvas with frame where there's drawing
        canvas_fg = cv2.bitwise_and(canvas_resized, canvas_resized, mask=mask)
        
        # For smooth blending on drawing areas
        if np.any(mask):
            # Convert mask to 3 channels for blending
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            # Blend only where there's drawing
            blended = frame.copy()
            drawing_area = mask_3ch > 0
            blended[drawing_area] = (
                frame[drawing_area] * (1 - alpha) + 
                canvas_resized[drawing_area] * alpha
            ).astype(np.uint8)
            
            return blended
        else:
            return frame
    
    def has_drawing(self) -> bool:
        """
        Check apakah ada drawing di canvas.
        
        Returns:
            True jika ada drawing, False jika kosong
        """
        return len(self.all_strokes) > 0


def main():
    print("Instructions:")
    print("- Hold SPACE and move mouse to draw")
    print("- Press 'c' to clear canvas")
    print("- Press 's' to show preprocessed digit")
    print("- Press 'q' to quit")
    print()
    
    # Initialize
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    
    # Get frame size
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read frame")
        return
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Initialize drawer
    drawer = AirDrawer(canvas_size=(h, w))
    
    # Mouse callback untuk testing
    drawing = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawer.start_drawing((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawer.add_point((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            drawer.end_drawing()
    
    cv2.namedWindow("Air Drawer Demo")
    cv2.setMouseCallback("Air Drawer Demo", mouse_callback)
    
    print("[INFO] Starting demo...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Overlay canvas
            result = drawer.overlay_on_frame(frame, alpha=0.8)
            
            # Show instructions
            cv2.putText(result, "Draw with LEFT MOUSE button", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, "Press 'c' to clear, 's' to show digit, 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Air Drawer Demo", result)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                drawer.clear_canvas()
            elif key == ord('s'):
                digit = drawer.get_preprocessed_digit()
                if digit is not None:
                    # Show preprocessed digit
                    digit_display = cv2.resize(digit, (280, 280), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Preprocessed Digit (28x28)", digit_display)
                    print("[INFO] Preprocessed digit shape:", digit.shape)
                else:
                    print("[WARNING] No drawing to process")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Demo ended")


if __name__ == "__main__":
    main()