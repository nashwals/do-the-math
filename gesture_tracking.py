"""
Modul Gesture Tracking untuk DO THE MATH
Menggunakan CVZone HandDetector dan Digit Recognizer
"""

import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from digit_recognition import load_model, recognize_multi_digit

# Konstanta
DRAW_CHARGE_TIME = 30
NOTIFICATION_DURATION = 30
BRUSH_SIZE = 20

# Inisialisasi
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, 
                        detectionCon=0.7, minTrackCon=0.5)

try:
    session = load_model()
    RECOGNIZER_LOADED = True
except:
    RECOGNIZER_LOADED = False
    session = None

# Variabel state
previousPosition = None
canvas = None
draw_charge_counter = 0
is_drawing_allowed = False
notification_text = ""
notification_timer = 0

def getHandInfo(img):
    """
    Mendapatkan informasi tangan dari frame.
    
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


def draw(info, previousPosition, canvas, img):
    """
    Menangani logika drawing berdasarkan gesture tangan.
    
    Args:
        info: Tuple (fingers, lmlist)
        previousPosition: Posisi sebelumnya untuk drawing
        canvas: Canvas untuk menggambar
        img: Frame gambar
        
    Returns:
        tuple: (currentPosition, canvas)
    """
    global draw_charge_counter, is_drawing_allowed, notification_text, notification_timer
    
    fingers, lmlist = info
    currentPosition = None
    
    # Mode Drawing: 1 jari (telunjuk)
    if fingers == [0, 1, 0, 0, 0]:
        currentPosition = lmlist[8][0:2]
        
        if not is_drawing_allowed:
            draw_charge_counter += 1
            if draw_charge_counter >= DRAW_CHARGE_TIME:
                is_drawing_allowed = True
                previousPosition = currentPosition
        
        if is_drawing_allowed:
            if previousPosition is None:
                previousPosition = currentPosition
            
            cv2.line(canvas, currentPosition, previousPosition, 
                    (255, 255, 255), BRUSH_SIZE)
            cv2.circle(canvas, currentPosition, 5, (255, 255, 255), cv2.FILLED)
            
            previousPosition = currentPosition
    
    # Mode Clear: 5 jari (tidak perlu tunggu notification timer)
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)
        if notification_timer == 0:
            notification_text = "Hapus Canvas"
            notification_timer = NOTIFICATION_DURATION
        
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    # Mode Submit dan Recognize: 4 jari (tanpa jempol)
    elif fingers == [0, 1, 1, 1, 1]:
        if notification_timer == 0:
            if RECOGNIZER_LOADED:
                result, confidence = recognize_multi_digit(session, canvas, max_digits=3)
                
                if result is not None and confidence > 50:
                    notification_text = f"ANGKA: {result}"
                    cv2.imwrite("hasil_gambar.png", canvas)
                else:
                    notification_text = "TIDAK JELAS!"
            else:
                notification_text = "MODEL NOT LOADED!"
            
            notification_timer = NOTIFICATION_DURATION
        
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    # Mode Idle
    else:
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    return currentPosition, canvas


def displayInstructions(img):
    """
    Menampilkan instruksi penggunaan di layar.
    """
    instructions = [
        "INSTRUKSI:",
        "1 Jari (telunjuk) = Draw",
        "4 Jari (tanpa jempol) = Submit & Recognize",
        "5 Jari = Clear Canvas",
        "ketik 'q' = Quit"
    ]
    
    y_offset = 30
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, y_offset + (i * 30)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def displayReadyButton(img):
    """
    Menampilkan status ready/drawing button di pojok kanan atas.
    """
    global draw_charge_counter, is_drawing_allowed
    
    button_x = img.shape[1] - 150
    button_y = 20
    button_w = 130
    button_h = 40
    
    if draw_charge_counter > 0 and not is_drawing_allowed:
        progress = draw_charge_counter / DRAW_CHARGE_TIME
        
        overlay = img.copy()
        cv2.rectangle(overlay, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        cv2.rectangle(img, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (0, 255, 255), 2)
        
        fill_w = int(button_w * progress)
        cv2.rectangle(img, (button_x, button_y), (button_x + fill_w, button_y + button_h),
                     (0, 255, 0), -1)
        
        cv2.putText(img, "READY...", (button_x + 10, button_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    elif is_drawing_allowed:
        overlay = img.copy()
        cv2.rectangle(overlay, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (128, 0, 128), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        cv2.rectangle(img, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (255, 0, 255), 2)
        
        cv2.putText(img, "DRAWING", (button_x + 10, button_y + 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def displayNotification(img):
    """
    Menampilkan notifikasi hasil recognition di pojok kanan bawah.
    """
    global notification_text, notification_timer
    
    if notification_timer > 0:
        text_x = img.shape[1] - 200
        text_y = img.shape[0] - 20
        
        if "ANGKA:" in notification_text:
            color = (0, 255, 0)
        elif "TIDAK JELAS" in notification_text or "NOT LOADED" in notification_text:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        
        cv2.putText(img, notification_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        notification_timer -= 1


def displayFingerStatus(img, fingers):
    """
    Menampilkan status jari yang terdeteksi di pojok kiri bawah.
    """
    if fingers:
        finger_text = f"Jari: {fingers}"
        cv2.putText(img, finger_text, (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# Main Loop
while True:
    success, img = cap.read()
    
    if not success:
        break
    
    img = cv2.flip(img, 1)
    
    if canvas is None:
        canvas = np.zeros_like(img)
    
    info = getHandInfo(img)
    fingers = None
    
    if info:
        fingers, lmlist = info
        previousPosition, canvas = draw(info, previousPosition, canvas, img)
        displayFingerStatus(img, fingers)
    else:
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    combinedImage = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    
    displayInstructions(combinedImage)
    
    if fingers is not None:
        displayReadyButton(combinedImage)
    
    displayNotification(combinedImage)
    
    cv2.imshow("DO THE MATH", combinedImage)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()