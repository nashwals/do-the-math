"""
Gesture Tracking Module untuk DO THE MATH!
==========================================

Modul ini mengelola:
1. Hand detection dan tracking menggunakan MediaPipe
2. Gesture recognition (1 jari, 4 jari, 5 jari)
3. Drawing canvas management
4. Gesture-based interactions
"""

import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from typing import Optional, Tuple


class GestureController:
    """
    Class untuk mengelola gesture tracking dan drawing canvas.
    
    Attributes:
        detector (HandDetector): MediaPipe hand detector
        draw_charge_time (int): Frame yang dibutuhkan untuk aktivasi drawing
        brush_size (int): Ukuran brush untuk menggambar
        draw_charge_counter (int): Counter untuk charging draw mode
        is_drawing_allowed (bool): Flag apakah drawing sudah aktif
        previous_position (tuple): Posisi finger landmark sebelumnya
    """
    
    def __init__(self, draw_charge_time: int = 30, brush_size: int = 20):
        """
        Inisialisasi Gesture Controller.
        
        Args:
            draw_charge_time (int): Frame yang diperlukan untuk aktivasi drawing
            brush_size (int): Ukuran brush untuk menggambar
        """
        # Initialize hand detector
        self.detector = HandDetector(
            staticMode=False,
            maxHands=1,
            modelComplexity=1,
            detectionCon=0.7,
            minTrackCon=0.5
        )
        
        # Gesture settings
        self.draw_charge_time = draw_charge_time
        self.brush_size = brush_size
        
        # Drawing state
        self.draw_charge_counter = 0
        self.is_drawing_allowed = False
        self.previous_position: Optional[Tuple[int, int]] = None
        
        print("✓ Gesture Controller initialized")
    
    def get_hand_info(self, img: np.ndarray) -> Optional[Tuple[list, list]]:
        """
        Mendeteksi tangan dan mengembalikan informasi jari dan landmark.
        
        Args:
            img (numpy.ndarray): Frame gambar dari webcam
            
        Returns:
            tuple: (fingers, lmList) - Status jari dan list landmark
            None: Jika tidak ada tangan yang terdeteksi
        """
        hands, img = self.detector.findHands(img, draw=True, flipType=True)
        
        if hands:
            hand1 = hands[0]
            lm_list = hand1["lmList"]  # List of 21 landmarks
            fingers = self.detector.fingersUp(hand1)  # Status jari (0=lipat, 1=tegak)
            return fingers, lm_list
        else:
            return None
    
    def process_gesture(
        self,
        info: Tuple[list, list],
        canvas: np.ndarray,
        img: np.ndarray,
        is_blocked: bool = False
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray, Optional[str]]:
        """
        Memproses gesture dan menggambar di canvas.
        
        Gesture modes:
        - [0,1,0,0,0]: Drawing mode (1 jari telunjuk)
        - [1,1,1,1,1]: Clear canvas (5 jari)
        - [0,1,1,1,1]: Submit answer (4 jari tanpa jempol)
        
        Args:
            info (tuple): (fingers, lmList) dari get_hand_info()
            canvas (numpy.ndarray): Canvas untuk menggambar
            img (numpy.ndarray): Frame gambar dari webcam
            is_blocked (bool): Flag untuk block gesture processing
            
        Returns:
            tuple: (current_position, updated_canvas, action)
                - current_position: Posisi finger saat ini atau None
                - updated_canvas: Canvas yang sudah diupdate
                - action: String action ("submit", "clear", None)
        """
        # Block gesture jika diminta
        if is_blocked:
            return self.previous_position, canvas, None
        
        fingers, lm_list = info
        current_position = None
        action = None
        
        # Mode menggambar: hanya jari telunjuk yang terangkat
        if fingers == [0, 1, 0, 0, 0]:
            current_position = tuple(lm_list[8][0:2])  # Posisi ujung jari telunjuk
            
            # Logika charging untuk aktivasi drawing
            if not self.is_drawing_allowed:
                self.draw_charge_counter += 1
                if self.draw_charge_counter >= self.draw_charge_time:
                    self.is_drawing_allowed = True
                    self.previous_position = current_position
            
            # Mulai menggambar setelah charging selesai
            if self.is_drawing_allowed:
                if self.previous_position is None:
                    self.previous_position = current_position
                
                # Gambar garis dari posisi sebelumnya ke posisi sekarang
                cv2.line(canvas, current_position, self.previous_position, 
                        (255, 255, 255), self.brush_size)
                
                # Gambar lingkaran kecil di posisi saat ini untuk smooth effect
                cv2.circle(canvas, current_position, 5, (255, 255, 255), cv2.FILLED)
                
                self.previous_position = current_position
        
        # Mode hapus: semua jari terangkat
        elif fingers == [1, 1, 1, 1, 1]:
            canvas = np.zeros_like(img)
            action = "clear"
            self._reset_drawing_state()
        
        # Mode submit: 4 jari tanpa jempol
        elif fingers == [0, 1, 1, 1, 1]:
            action = "submit"
            self._reset_drawing_state()
        
        # Mode idle: reset drawing state
        else:
            self._reset_drawing_state()
        
        return current_position, canvas, action
    
    def _reset_drawing_state(self):
        """
        Reset drawing state ke kondisi awal.
        
        Internal method untuk cleanup state saat mode berubah.
        """
        self.previous_position = None
        self.is_drawing_allowed = False
        self.draw_charge_counter = 0
    
    def get_draw_progress(self) -> float:
        """
        Mendapatkan progress charging untuk drawing mode.
        
        Returns:
            float: Progress 0.0 - 1.0
        """
        return min(1.0, self.draw_charge_counter / self.draw_charge_time)
    
    def is_ready_to_draw(self) -> bool:
        """
        Check apakah drawing mode sudah aktif.
        
        Returns:
            bool: True jika sudah bisa menggambar
        """
        return self.is_drawing_allowed
    
    def reset(self):
        """
        Reset semua state gesture controller.
        
        Digunakan saat restart quiz atau clear semua state.
        """
        self._reset_drawing_state()
        print("✓ Gesture controller reset")


# Test function
if __name__ == "__main__":
    print("Testing Gesture Controller Module...")
    print("-" * 60)
    
    try:
        # Initialize controller
        controller = GestureController(draw_charge_time=30, brush_size=20)
        
        print("\n✓ Gesture Controller initialized successfully")
        print(f"  - Draw charge time: {controller.draw_charge_time} frames")
        print(f"  - Brush size: {controller.brush_size} pixels")
        print(f"  - Drawing allowed: {controller.is_drawing_allowed}")
        
        # Test reset
        controller.reset()
        print("\n✓ Reset function works")
        
        print("\n" + "="*60)
        print("✓ Gesture Controller Module Test: PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()