import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector


class GestureTracking:
    """
    Class untuk menangani gesture detection dan air drawing
    """
    
    def __init__(self, width, height, brush_size=20, draw_charge_time=30):
        """
        Initialize GestureHandler
        
        Args:
            width: Lebar canvas
            height: Tinggi canvas
            brush_size: Ukuran brush untuk drawing
            draw_charge_time: Frames yang dibutuhkan untuk charge drawing mode
        """
        self.width = width
        self.height = height
        self.brush_size = brush_size
        self.draw_charge_time = draw_charge_time
        
        # Hand detector
        self.detector = HandDetector(
            staticMode=False,
            maxHands=1,
            modelComplexity=1,
            detectionCon=0.7,
            minTrackCon=0.5
        )
        
        # Drawing state
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.previous_position = None
        self.is_drawing_allowed = False
        self.draw_charge_counter = 0
        
    def get_hand_info(self, img):
        """
        Mendapatkan informasi tangan dari frame.
        
        Args:
            img: Frame gambar dari webcam
            
        Returns:
            tuple: (fingers, lmList) atau None jika tidak ada tangan
        """
        hands, img = self.detector.findHands(img, draw=True, flipType=True)
        
        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = self.detector.fingersUp(hand)
            return fingers, lmList
        return None
    
    def handle_gesture(self, img, info):
        """
        Menangani logika gesture berdasarkan jari yang terdeteksi
        
        Args:
            img: Frame gambar
            info: Tuple (fingers, lmlist) dari get_hand_info()
            
        Returns:
            dict: Status gesture dengan keys:
                - 'action': 'draw', 'submit', 'clear', 'idle'
                - 'charging': bool (sedang charging atau tidak)
                - 'ready': bool (ready untuk draw atau tidak)
        """
        if info is None:
            self.previous_position = None
            # JANGAN reset counter dan is_drawing_allowed
            # Biar tetap ready walau tangan hilang sebentar
            return {
                'action': 'idle',
                'charging': False,
                'ready': self.is_drawing_allowed
            }
        
        fingers, lmlist = info
        
        # Mode Drawing: 1 jari (telunjuk)
        if fingers == [0, 1, 0, 0, 0]:
            current_pos = lmlist[8][0:2]
            
            if not self.is_drawing_allowed:
                self.draw_charge_counter += 1
                
                if self.draw_charge_counter >= self.draw_charge_time:
                    self.is_drawing_allowed = True
                    self.previous_position = current_pos
                
                return {
                    'action': 'charging',
                    'charging': True,
                    'ready': False,
                    'progress': self.draw_charge_counter / self.draw_charge_time
                }
            
            # Drawing aktif
            if self.previous_position is None:
                self.previous_position = current_pos
            
            cv2.line(self.canvas, current_pos, self.previous_position,
                    (255, 255, 255), self.brush_size)
            cv2.circle(self.canvas, current_pos, 5, (255, 255, 255), cv2.FILLED)
            
            self.previous_position = current_pos
            
            return {
                'action': 'draw',
                'charging': False,
                'ready': True
            }
        
        # Mode Clear: 5 jari
        elif fingers == [1, 1, 1, 1, 1]:
            self.clear_canvas()
            
            return {
                'action': 'clear',
                'charging': False,
                'ready': False
            }
        
        # Mode Submit: 4 jari (tanpa jempol)
        elif fingers == [0, 1, 1, 1, 1]:
            return {
                'action': 'submit',
                'charging': False,
                'ready': False
            }
        
        # Gesture lain (idle)
        else:
            self.previous_position = None
            # JANGAN reset is_drawing_allowed dan draw_charge_counter
            
            return {
                'action': 'idle',
                'charging': False,
                'ready': self.is_drawing_allowed
            }
    
    def clear_canvas(self):
        """Clear drawing canvas dan reset state"""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.is_drawing_allowed = False
        self.draw_charge_counter = 0
        self.previous_position = None
    
    def get_canvas(self):
        """Get current canvas untuk display atau recognition"""
        return self.canvas.copy()
    
    def reset_drawing_state(self):
        """Reset drawing state setelah submit"""
        self.is_drawing_allowed = False
        self.draw_charge_counter = 0
        self.previous_position = None