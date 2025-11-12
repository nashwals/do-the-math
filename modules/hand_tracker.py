import cv2
import mediapipe as mp
import time
from enum import Enum
from typing import Optional, Tuple, List


class GestureType(Enum):
    """Enum untuk tipe gesture yang dikenali sistem."""
    NONE = 0
    DRAWING = 1  # Hanya telunjuk
    ERASE = 2    # 5 jari terbuka
    SUBMIT = 3   # 4 jari terbuka 


class HandTracker:
    """
    Class untuk tracking tangan dan mengenali gesture menggunakan MediaPipe.
    
    Attributes:
        mp_hands: MediaPipe Hands solution
        mp_draw: MediaPipe drawing utilities
        hands: Configured hands detector
        brush_delay: Waktu delay sebelum brush muncul (detik)
        gesture_history: Buffer untuk gesture stability
    """
    
    # Finger tip && pip landmark IDs (MediaPipe Hand Landmarks)
    THUMB_TIP, THUMB_IP = 4, 3
    INDEX_TIP, INDEX_PIP = 8, 6
    MIDDLE_TIP, MIDDLE_PIP = 12, 10
    RING_TIP, RING_PIP = 16, 14
    PINKY_TIP, PINKY_PIP = 20, 18
    
    def __init__(self, 
                 max_hands: int = 5,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.7,
                 brush_delay: float = 1.5):
        """
        Initialize Hand Tracker.
        
        Args:
            max_hands: Maximum jumlah tangan yang dideteksi
            detection_confidence: Minimum confidence untuk hand detection
            tracking_confidence: Minimum confidence untuk hand tracking
            brush_delay: Delay dalam detik sebelum brush muncul (default: 1.5 detik)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        
        # Configure MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Brush delay timer
        self.brush_delay = brush_delay
        self.index_start_time = None
        self.brush_active = False
        
        # Gesture stability buffer (untuk menghindari false detection)
        self.gesture_history = []
        self.gesture_buffer_size = 5
        
        print(f"[HandTracker] Initialized with max_hands={max_hands}, "
              f"brush_delay={brush_delay}s")
    
    def detect_hands(self, frame: cv2.Mat) -> Optional[object]:
        """
        Deteksi tangan pada frame menggunakan MediaPipe.
        
        Args:
            frame: Input frame dari webcam (BGR format)
            
        Returns:
            MediaPipe hands results object atau None jika tidak ada tangan terdeteksi
        """
        # Convert BGR to RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(frame_rgb)
        
        return results
    
    def is_finger_up(self, landmarks, finger_tip_id: int, finger_pip_id: int) -> bool:
        """
        Check apakah jari tertentu sedang terangkat.
        
        Args:
            landmarks: Hand landmarks dari MediaPipe
            finger_tip_id: ID landmark untuk ujung jari
            finger_pip_id: ID landmark untuk PIP joint
            
        Returns:
            True jika jari terangkat, False jika tidak
        """
        # Jari dianggap terangkat jika tip berada di atas PIP (y lebih kecil)
        # Tambahkan threshold untuk mengurangi false detection
        tip_y = landmarks[finger_tip_id].y
        pip_y = landmarks[finger_pip_id].y
        
        # Jari harus lebih tinggi minimal 0.02 (2% dari tinggi frame)
        return (pip_y - tip_y) > 0.02
    
    def is_thumb_up(self, landmarks) -> bool:
        """
        Check apakah jempol terangkat (logika berbeda karena orientasi horizontal).
        
        Args:
            landmarks: Hand landmarks dari MediaPipe
            
        Returns:
            True jika jempol terangkat, False jika tidak
        """
        # Untuk jempol, check berdasarkan posisi x dan y
        thumb_tip = landmarks[self.THUMB_TIP]
        thumb_ip = landmarks[self.THUMB_IP]
        wrist = landmarks[0]
        
        # Jempol terangkat jika tip lebih jauh dari wrist dibanding IP
        thumb_tip_dist = abs(thumb_tip.x - wrist.x) + abs(thumb_tip.y - wrist.y)
        thumb_ip_dist = abs(thumb_ip.x - wrist.x) + abs(thumb_ip.y - wrist.y)
        
        return thumb_tip_dist > thumb_ip_dist
    
    def detect_gesture(self, landmarks) -> GestureType:
        """
        Deteksi gesture berdasarkan konfigurasi jari.
        Priority: ERASE (5) > SUBMIT > DRAWING (telunjuk)
        """
        thumb = self.is_thumb_up(landmarks)
        index = self.is_finger_up(landmarks, self.INDEX_TIP, self.INDEX_PIP)
        middle = self.is_finger_up(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring = self.is_finger_up(landmarks, self.RING_TIP, self.RING_PIP)
        pinky = self.is_finger_up(landmarks, self.PINKY_TIP, self.PINKY_PIP)
        
        count = sum([thumb, index, middle, ring, pinky])
        
        # Priority detection
        if count >= 5:
            return GestureType.ERASE
        if count >= 4:
            return GestureType.SUBMIT
        if index and not middle and not ring and not pinky and not thumb:
            return GestureType.DRAWING
        
        return GestureType.NONE
    
    def get_stable_gesture(self, current_gesture: GestureType) -> GestureType:
        """
        Stabilkan gesture detection dengan buffer untuk menghindari false positives.
        
        Args:
            current_gesture: Gesture yang terdeteksi saat ini
            
        Returns:
            Stabilized GestureType
        """
        self.gesture_history.append(current_gesture)
        
        if len(self.gesture_history) > self.gesture_buffer_size:
            self.gesture_history.pop(0)
        
        # Butuh minimal 3 frame untuk stability
        if len(self.gesture_history) < 3:
            return GestureType.NONE
        
        # Count occurrences
        gesture_counts = {}
        for gesture in self.gesture_history:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Return most common gesture (simple majority)
        max_gesture = max(gesture_counts, key=gesture_counts.get)
        max_count = gesture_counts[max_gesture]
        
        # Need at least 50% of buffer
        if max_count >= len(self.gesture_history) * 0.5:
            return max_gesture
        
        return GestureType.NONE
    
    def get_index_finger_tip(self, landmarks, frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Dapatkan koordinat pixel ujung jari telunjuk.
        
        Args:
            landmarks: Hand landmarks dari MediaPipe
            frame_shape: (height, width) dari frame
            
        Returns:
            Tuple (x, y) koordinat pixel dari ujung telunjuk
        """
        height, width = frame_shape
        index_tip = landmarks[self.INDEX_TIP]
        
        # Convert normalized coordinates (0-1) to pixel coordinates
        x = int(index_tip.x * width)
        y = int(index_tip.y * height)
        
        return (x, y)
    
    def update_brush_state(self, gesture: GestureType) -> bool:
        """
        Update status brush berdasarkan gesture dan delay timer.
        
        Brush akan muncul setelah telunjuk dipertahankan selama brush_delay detik.
        
        Args:
            gesture: Current gesture type
            
        Returns:
            True jika brush aktif, False jika tidak
        """
        if gesture == GestureType.DRAWING:
            # Start timer if not started
            if self.index_start_time is None:
                self.index_start_time = time.time()
            
            # Check if delay has passed
            elapsed = time.time() - self.index_start_time
            if elapsed >= self.brush_delay:
                self.brush_active = True
            else:
                self.brush_active = False
        else:
            # Reset timer and brush state
            self.index_start_time = None
            self.brush_active = False
        
        return self.brush_active
    
    def get_brush_timer_progress(self) -> float:
        """
        Dapatkan progress timer brush (0.0 - 1.0).
        
        Returns:
            Progress dari 0.0 (baru mulai) hingga 1.0 (brush ready)
        """
        if self.index_start_time is None:
            return 0.0
        
        elapsed = time.time() - self.index_start_time
        progress = min(elapsed / self.brush_delay, 1.0)
        
        return progress
    
    def draw_brush_indicator(self, frame: cv2.Mat, position: Tuple[int, int], 
                            progress: float) -> None:
        """
        Gambar indikator brush di ujung telunjuk dengan progress ring.
        
        Args:
            frame: Frame untuk digambar (modified in-place)
            position: (x, y) posisi brush
            progress: Progress timer (0.0 - 1.0)
        """
        x, y = position
        
        if progress < 1.0:
            # Draw progress ring
            radius = 20
            angle = int(360 * progress)
            
            # Background circle (gray)
            cv2.circle(frame, (x, y), radius, (200, 200, 200), 2)
            
            # Progress arc (yellow)
            if angle > 0:
                cv2.ellipse(frame, (x, y), (radius, radius), -90, 0, angle, 
                           (0, 255, 255), 3)
            
            # Center dot
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        else:
            # Draw full brush (red)
            cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)
            cv2.circle(frame, (x, y), 18, (0, 0, 255), 2)
    
    def draw_gesture_info(self, frame: cv2.Mat, gesture: GestureType, 
                         position: Tuple[int, int] = (10, 30)) -> None:
        """
        Tampilkan informasi gesture saat ini pada frame.
        
        Args:
            frame: Frame untuk digambar (modified in-place)
            gesture: Current gesture type
            position: (x, y) posisi text
        """
        gesture_text = {
            GestureType.NONE: "No Gesture",
            GestureType.DRAWING: "DRAWING MODE",
            GestureType.ERASE: "ERASE MODE",
            GestureType.SUBMIT: "SUBMIT ANSWER"
        }
        
        gesture_color = {
            GestureType.NONE: (200, 200, 200),
            GestureType.DRAWING: (0, 255, 0),
            GestureType.ERASE: (0, 165, 255),
            GestureType.SUBMIT: (255, 0, 255)
        }
        
        text = gesture_text.get(gesture, "Unknown")
        color = gesture_color.get(gesture, (255, 255, 255))
        
        # Draw background rectangle
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, 
                     (position[0] - 5, position[1] - text_size[1] - 5),
                     (position[0] + text_size[0] + 5, position[1] + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2, cv2.LINE_AA)
    
    def release(self) -> None:
        """
        Release MediaPipe resources.
        """
        self.hands.close()
        print("[HandTracker] Resources released")


def main():
    print("Gestures:")
    print("1. Index finger only -> DRAWING MODE (brush muncul setelah 1.5 detik)")
    print("2. All 5 fingers open -> ERASE MODE")
    print("3. 4 fingers (thumb, index, middle, ring) -> SUBMIT")
    print("Press 'q' to quit")
    print()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    
    # Initialize hand tracker
    tracker = HandTracker(brush_delay=1.5)
    
    print("[INFO] Starting camera feed...")
    
    # Variables untuk tracking gesture actions
    last_erase_time = 0
    last_submit_time = 0
    action_cooldown = 1.0  # Cooldown 1 detik untuk aksi
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("[ERROR] Failed to read frame")
                break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = tracker.detect_hands(frame)
            
            # Process if hand detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    landmarks = hand_landmarks.landmark
                    
                    # Detect gesture
                    current_gesture = tracker.detect_gesture(landmarks)
                    stable_gesture = tracker.get_stable_gesture(current_gesture)
                    
                    # Update brush state
                    brush_active = tracker.update_brush_state(stable_gesture)
                    
                    # Handle gestures
                    if stable_gesture == GestureType.DRAWING:
                        finger_pos = tracker.get_index_finger_tip(
                            landmarks, 
                            (frame.shape[0], frame.shape[1])
                        )
                        
                        progress = tracker.get_brush_timer_progress()
                        tracker.draw_brush_indicator(frame, finger_pos, progress)
                        
                        if brush_active:
                            cv2.putText(frame, "Drawing Ready!", (finger_pos[0] + 25, finger_pos[1]), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    elif stable_gesture == GestureType.ERASE:
                        current_time = time.time()
                        if current_time - last_erase_time > action_cooldown:
                            cv2.putText(frame, "ERASING...", (frame.shape[1]//2 - 100, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                            print("[ACTION] Erasing canvas...")
                            last_erase_time = current_time
                    
                    elif stable_gesture == GestureType.SUBMIT:
                        current_time = time.time()
                        if current_time - last_submit_time > action_cooldown:
                            cv2.putText(frame, "SUBMITTING...", (frame.shape[1]//2 - 120, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
                            print("[ACTION] Jawaban anda sudah terkirim")
                            last_submit_time = current_time
                    
                    # Draw gesture info
                    tracker.draw_gesture_info(frame, stable_gesture)
            
            # Show frame
            cv2.imshow("Hand Tracker Demo - DO THE MATH!", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()
        print("[INFO] Demo ended")


if __name__ == "__main__":
    main()