"""
Modul Gesture Tracking untuk DO THE MATH
Library untuk hand gesture detection dan drawing
"""

import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

# ======================= KONSTANTA =======================
DRAW_CHARGE_TIME = 30
NOTIFICATION_DURATION = 60
BRUSH_SIZE = 20

# Dimensi window
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Area drawing
DRAWING_AREA_X = 300
DRAWING_AREA_WIDTH = WINDOW_WIDTH - DRAWING_AREA_X


# ======================= FACTORY FUNCTIONS =======================
def create_detector():
    """
    Factory function untuk membuat HandDetector instance
    
    Returns:
        HandDetector: Configured hand detector
    """
    return HandDetector(
        staticMode=False,
        maxHands=1,
        modelComplexity=1,
        detectionCon=0.7,
        minTrackCon=0.5
    )


def create_gesture_state():
    """
    Factory function untuk membuat gesture state dictionary
    
    Returns:
        dict: Initial gesture state
    """
    return {
        'previousPosition': None,
        'draw_charge_counter': 0,
        'is_drawing_allowed': False
    }


# ======================= HAND DETECTION =======================
def getHandInfo(detector, img):
    """
    Mendapatkan informasi tangan dari frame
    
    Args:
        detector: HandDetector instance
        img: Frame gambar
        
    Returns:
        tuple: (fingers, lmList) atau None jika tidak ada tangan terdeteksi
    """
    hands, img = detector.findHands(img, draw=True, flipType=True)
    
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]
        fingers = detector.fingersUp(hand1)
        return fingers, lmList
    else:
        return None


# ======================= GESTURE PROCESSING =======================
def process_gesture(info, state, canvas, img, brush_size=BRUSH_SIZE):
    """
    Memproses gesture tangan untuk drawing
    
    Args:
        info: Tuple (fingers, lmlist) dari getHandInfo
        state: Dictionary dengan gesture state
        canvas: Canvas untuk menggambar
        img: Frame gambar
        brush_size: Ukuran brush untuk drawing
        
    Returns:
        tuple: (canvas, action, updated_state)
            action: 'draw', 'clear', 'submit', atau None
    """
    fingers, lmlist = info
    action = None
    
    # Mode Drawing: 1 jari (telunjuk)
    if fingers == [0, 1, 0, 0, 0]:
        currentPosition = lmlist[8][0:2]
        
        if not state['is_drawing_allowed']:
            state['draw_charge_counter'] += 1
            if state['draw_charge_counter'] >= DRAW_CHARGE_TIME:
                state['is_drawing_allowed'] = True
                state['previousPosition'] = currentPosition
        
        if state['is_drawing_allowed']:
            if state['previousPosition'] is None:
                state['previousPosition'] = currentPosition
            
            cv2.line(canvas, currentPosition, state['previousPosition'],
                    (255, 255, 255), brush_size)
            cv2.circle(canvas, currentPosition, 5, (255, 255, 255), cv2.FILLED)
            
            state['previousPosition'] = currentPosition
            action = 'draw'
    
    # Mode Clear: 5 jari
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)
        state['previousPosition'] = None
        state['is_drawing_allowed'] = False
        state['draw_charge_counter'] = 0
        action = 'clear'
    
    # Mode Submit: 4 jari (tanpa jempol)
    elif fingers == [0, 1, 1, 1, 1]:
        state['previousPosition'] = None
        state['is_drawing_allowed'] = False
        state['draw_charge_counter'] = 0
        action = 'submit'
    
    # Mode Idle
    else:
        state['previousPosition'] = None
        state['is_drawing_allowed'] = False
        state['draw_charge_counter'] = 0
    
    return canvas, action, state


# ======================= UI DISPLAY FUNCTIONS =======================
def draw_status_indicator(img, state, x=20, y=500):
    """
    Menggambar indikator status drawing
    
    Args:
        img: Frame gambar
        state: Dictionary dengan gesture state
        x: Posisi x indikator
        y: Posisi y indikator
    """
    indicator_w = 260
    indicator_h = 40
    
    if state['draw_charge_counter'] > 0 and not state['is_drawing_allowed']:
        # Charging
        progress = state['draw_charge_counter'] / DRAW_CHARGE_TIME
        
        cv2.rectangle(img, (x, y),
                     (x + indicator_w, y + indicator_h),
                     (0, 255, 255), 2)
        
        fill_w = int(indicator_w * progress)
        cv2.rectangle(img, (x, y),
                     (x + fill_w, y + indicator_h),
                     (0, 255, 0), -1)
        
        cv2.putText(img, "READY...", (x + 70, y + 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    elif state['is_drawing_allowed']:
        # Drawing mode active
        cv2.rectangle(img, (x, y),
                     (x + indicator_w, y + indicator_h),
                     (255, 0, 255), -1)
        
        cv2.rectangle(img, (x, y),
                     (x + indicator_w, y + indicator_h),
                     (255, 255, 255), 2)
        
        cv2.putText(img, "DRAWING", (x + 60, y + 27),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)