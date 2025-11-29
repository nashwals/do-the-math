"""
Modul Gesture Tracking untuk DO THE MATH
Menggunakan CVZone HandDetector dan Digit Recognizer
DENGAN INTRO SCREEN YANG KEREN!
"""

import numpy as np
import cv2
import time
import math
import pygame
import os
from cvzone.HandTrackingModule import HandDetector
from digit_recognition import load_model, recognize_multi_digit

# ==================== KONSTANTA ====================
DRAW_CHARGE_TIME = 30
NOTIFICATION_DURATION = 30
BRUSH_SIZE = 20

# ==================== INISIALISASI ====================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Ambil resolusi asli dari webcam
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolusi kamera: {width}x{height}")

# Hitung faktor scaling untuk responsive design
scale_x = width / 1280.0
scale_y = height / 720.0
min_scale = min(scale_x, scale_y)

print(f"Faktor skala - X: {scale_x:.2f}, Y: {scale_y:.2f}")

# Inisialisasi pygame untuk audio
pygame.mixer.init()

# Load audio intro (jika ada)
intro_audio_path = "assets/audio/opening.wav"
audio_loaded = False
if os.path.exists(intro_audio_path):
    try:
        pygame.mixer.music.load(intro_audio_path)
        audio_loaded = True
        print("Audio intro berhasil dimuat!")
    except Exception as e:
        print(f"Error loading audio: {e}")
        audio_loaded = False
else:
    print(f"File audio tidak ditemukan: {intro_audio_path}")
    audio_loaded = False

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, 
                        detectionCon=0.7, minTrackCon=0.5)

try:
    session = load_model()
    RECOGNIZER_LOADED = True
    print("Model digit recognition berhasil dimuat")
except Exception as e:
    RECOGNIZER_LOADED = False
    session = None
    print(f"Error loading model: {e}")

# ==================== VARIABEL STATE ====================
previousPosition = None
canvas = None
draw_charge_counter = 0
is_drawing_allowed = False
notification_text = ""
notification_timer = 0

# State untuk intro screen
show_intro = True  # Mulai dengan intro screen
audio_played = False  # Flag untuk audio sudah diplay atau belum


# ==================== HELPER FUNCTIONS ====================

def get_scaled_font(base_size):
    """Mengambil ukuran font yang sudah discale"""
    return max(0.4, base_size * min_scale)


def get_scaled_thickness(base_thickness):
    """Mengambil ketebalan garis yang sudah discale"""
    return max(1, int(base_thickness * min_scale))


# ==================== INTRO SCREEN FUNCTION ====================

def draw_attract_mode(img):
    """
    Fungsi untuk menggambar layar intro game (ATTRACT MODE)
    
    Fitur yang ada di intro screen ini:
    - Overlay gelap dengan efek gradient
    - Background pattern grid yang subtle
    - Judul utama dengan efek bayangan
    - Kotak instruksi dengan transparansi
    - Icon gesture (satu jari, empat jari, lima jari)
    - Tombol "Tekan SPACE untuk Mulai" yang berkedip
    
    Filosofi desain:
    - UI yang clean dan modern
    - Kontras tinggi biar gampang dibaca
    - Responsive scaling buat berbagai resolusi layar
    """
    # Bikin overlay gelap di atas video
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    # Bikin background pattern grid yang halus
    pattern_overlay = img.copy()
    grid_spacing = int(80 * min_scale)
    
    # Gambar garis vertikal
    for x in range(0, width, grid_spacing):
        cv2.line(pattern_overlay, (x, 0), (x, height), (30, 30, 30), 1)
    
    # Gambar garis horizontal
    for y in range(0, height, grid_spacing):
        cv2.line(pattern_overlay, (0, y), (width, y), (30, 30, 30), 1)
    
    # Blend pattern dengan opacity rendah biar ga terlalu kelihatan
    cv2.addWeighted(pattern_overlay, 0.15, img, 0.85, 0, img)
    
    # Gambar judul utama dengan efek shadow
    title = "DO THE MATH!"
    title_font = cv2.FONT_HERSHEY_TRIPLEX  # Font khusus buat judul
    title_font_scale = get_scaled_font(3.5)
    title_thickness = get_scaled_thickness(7)
    
    # Hitung posisi judul biar di tengah atas
    title_size = cv2.getTextSize(title, title_font, title_font_scale, title_thickness)[0]
    title_x = (width - title_size[0]) // 2
    title_y = int(height * 0.22)
    
    # Gambar bayangan judul dulu (warna hitam, offset dikit)
    shadow_offset = int(4 * min_scale)
    cv2.putText(img, title, (title_x + shadow_offset, title_y + shadow_offset), 
               title_font, title_font_scale, (0, 0, 0), title_thickness + 2)
    
    # Gambar judul asli di atas bayangan (warna cyan/kuning sesuai foto)
    cv2.putText(img, title, (title_x, title_y), 
               title_font, title_font_scale, (0, 255, 255), title_thickness)
    
    # Sisanya pakai font SIMPLEX yang lebih jelas
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Gambar subtitle atau tagline
    subtitle = "Kuis Matematika - Gambar Jawabanmu di Udara!"
    subtitle_font_scale = get_scaled_font(0.9)
    subtitle_thickness = get_scaled_thickness(3)
    
    subtitle_size = cv2.getTextSize(subtitle, font, subtitle_font_scale, subtitle_thickness)[0]
    subtitle_x = (width - subtitle_size[0]) // 2
    subtitle_y = int(height * 0.30)
    
    cv2.putText(img, subtitle, (subtitle_x, subtitle_y), 
               font, subtitle_font_scale, (255, 255, 255), subtitle_thickness)
    
    # Bikin kotak instruksi dengan transparansi
    box_width = int(width * 0.6)
    box_height = int(height * 0.35)
    box_x = (width - box_width) // 2
    box_y = int(height * 0.38)
    
    # Gambar kotak dengan transparansi
    box_overlay = img.copy()
    cv2.rectangle(box_overlay, (box_x, box_y), 
                 (box_x + box_width, box_y + box_height),
                 (50, 50, 50), -1)
    cv2.addWeighted(box_overlay, 0.7, img, 0.3, 0, img)
    
    # Gambar border kotak (warna kuning sesuai foto)
    cv2.rectangle(img, (box_x, box_y), 
                 (box_x + box_width, box_y + box_height),
                 (0, 255, 255), 3)
    
    # Gambar judul instruksi
    inst_font_scale = get_scaled_font(0.7)
    inst_thickness = get_scaled_thickness(3)
    
    inst_title = "CARA BERMAIN:"
    inst_title_size = cv2.getTextSize(inst_title, font, inst_font_scale + 0.1, inst_thickness + 1)[0]
    inst_title_x = box_x + (box_width - inst_title_size[0]) // 2
    inst_title_y = box_y + int(box_height * 0.15)
    
    cv2.putText(img, inst_title, (inst_title_x, inst_title_y),
               font, inst_font_scale + 0.1, (0, 255, 255), inst_thickness + 1)
    
    # Daftar instruksi (sesuai foto)
    instructions = [
        ("assets/icons/one.png", "1 jari = Menggambar angka"),
        ("assets/icons/two.png", "4 jari = Submit jawaban"),
        ("assets/icons/five.png", "5 jari = Hapus gambar"),
    ]
    
    text_x = box_x + int(box_width * 0.35)
    start_y = inst_title_y + int(box_height * 0.2)
    line_spacing = int(box_height * 0.18)
    icon_size = int(40 * min_scale)
    
    for i, (icon_path, text) in enumerate(instructions):
        y_pos = start_y + (i * line_spacing)
        
        # Coba load dan tampilkan icon gambar
        icon_x = text_x - icon_size - int(15 * min_scale)
        icon_y = y_pos - icon_size + int(5 * min_scale)
        
        try:
            if os.path.exists(icon_path):
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon_resized = cv2.resize(icon, (icon_size, icon_size))
                    
                    # Handle transparansi icon (alpha channel)
                    if icon_resized.shape[2] == 4:
                        alpha = icon_resized[:, :, 3] / 255.0
                        for c in range(3):
                            img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c] = \
                                alpha * icon_resized[:, :, c] + \
                                (1 - alpha) * img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c]
                    else:
                        img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = icon_resized
                else:
                    # Fallback ke emoji kalau gambar gagal load
                    emoji_icons = ["☝️", "✌️", "✋"]
                    cv2.putText(img, emoji_icons[i], (icon_x, y_pos),
                               font, inst_font_scale * 0.8, (255, 255, 255), inst_thickness)
            else:
                # Fallback ke emoji kalau file tidak ada
                emoji_icons = ["☝️", "✌️", "✋"]
                cv2.putText(img, emoji_icons[i], (icon_x, y_pos),
                           font, inst_font_scale * 0.8, (255, 255, 255), inst_thickness)
        except Exception as e:
            # Fallback ke emoji kalau ada error
            emoji_icons = ["☝️", "✌️", "✋"]
            cv2.putText(img, emoji_icons[i], (icon_x, y_pos),
                       font, inst_font_scale * 0.8, (255, 255, 255), inst_thickness)
        
        # Gambar teks instruksi
        cv2.putText(img, text, (text_x, y_pos),
                   font, inst_font_scale, (255, 255, 255), inst_thickness)
    
    # Gambar call to action yang berkedip (hijau sesuai foto)
    cta_text = "[ Tekan SPACE untuk Mulai! ]"
    cta_font_scale = get_scaled_font(1.2)
    cta_thickness = get_scaled_thickness(4)
    
    # Efek berkedip pakai sine wave
    pulse = abs(math.sin(time.time() * 2)) * 0.3 + 0.7  # Range: 0.7 sampe 1.0
    cta_color = (int(0 * pulse), int(255 * pulse), int(0 * pulse))  # Hijau
    
    cta_size = cv2.getTextSize(cta_text, font, cta_font_scale, cta_thickness)[0]
    cta_x = (width - cta_size[0]) // 2
    cta_y = int(height * 0.85)
    
    cv2.putText(img, cta_text, (cta_x, cta_y),
               font, cta_font_scale, cta_color, cta_thickness)


# ==================== GESTURE FUNCTIONS ====================

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
                    print(f"Hasil recognition: {result} (confidence: {confidence:.2f}%)")
                else:
                    notification_text = "TIDAK JELAS!"
                    print("Digit tidak terdeteksi jelas")
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


# ==================== UI DISPLAY FUNCTIONS ====================

def displayInstructions(img):
    """
    Menampilkan instruksi penggunaan di layar.
    """
    instructions = [
        "INSTRUKSI:",
        "1 Jari (telunjuk) = Draw",
        "4 Jari (tanpa jempol) = Submit & Recognize",
        "5 Jari = Clear Canvas",
        "SPACE = Intro | Q = Quit"
    ]
    
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = get_scaled_font(0.5)
    thickness = get_scaled_thickness(2)
    
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, y_offset + (i * 30)), 
                   font, font_scale, (255, 255, 255), thickness)


def displayReadyButton(img):
    """
    Menampilkan status ready/drawing button di pojok kanan atas.
    """
    global draw_charge_counter, is_drawing_allowed
    
    button_x = width - 150
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
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = get_scaled_font(0.6)
        thickness = get_scaled_thickness(2)
        cv2.putText(img, "READY...", (button_x + 10, button_y + 28),
                   font, font_scale, (255, 255, 255), thickness)
    
    elif is_drawing_allowed:
        overlay = img.copy()
        cv2.rectangle(overlay, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (128, 0, 128), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        cv2.rectangle(img, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (255, 0, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = get_scaled_font(0.6)
        thickness = get_scaled_thickness(2)
        cv2.putText(img, "DRAWING", (button_x + 10, button_y + 28),
                   font, font_scale, (255, 255, 255), thickness)


def displayNotification(img):
    """
    Menampilkan notifikasi hasil recognition di pojok kanan bawah.
    """
    global notification_text, notification_timer
    
    if notification_timer > 0:
        text_x = width - 300
        text_y = height - 20
        
        if "ANGKA:" in notification_text:
            color = (0, 255, 0)
        elif "TIDAK JELAS" in notification_text or "NOT LOADED" in notification_text:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = get_scaled_font(0.7)
        thickness = get_scaled_thickness(2)
        
        cv2.putText(img, notification_text, (text_x, text_y), 
                   font, font_scale, color, thickness)
        notification_timer -= 1


def displayFingerStatus(img, fingers):
    """
    Menampilkan status jari yang terdeteksi di pojok kiri bawah.
    """
    if fingers:
        finger_text = f"Jari: {fingers}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = get_scaled_font(0.6)
        thickness = get_scaled_thickness(2)
        
        cv2.putText(img, finger_text, (10, height - 20), 
                   font, font_scale, (0, 255, 255), thickness)


# ==================== MAIN LOOP ====================

print("\n" + "="*50)
print("DO THE MATH - Digit Recognition dengan Gesture")
print("="*50)
print("Tekan SPACE untuk toggle intro screen")
print("Tekan Q untuk keluar")
print("="*50 + "\n")

def play_intro_audio():
    """Helper function untuk play audio intro"""
    global audio_played
    if audio_loaded and not audio_played:
        try:
            pygame.mixer.music.play()
            audio_played = True
            print("Playing intro audio...")
        except Exception as e:
            print(f"Error playing audio: {e}")

def toggle_intro_mode():
    """Helper function untuk toggle intro screen"""
    global show_intro, audio_played
    show_intro = not show_intro
    if show_intro:
        print("Menampilkan intro screen...")
        # Reset flag audio untuk diplay lagi
        audio_played = False
    else:
        print("Kembali ke mode drawing...")
        # Stop audio kalau lagi main
        if audio_loaded:
            pygame.mixer.music.stop()

while True:
    success, img = cap.read()
    
    if not success:
        print("Gagal membaca dari kamera!")
        break
    
    img = cv2.flip(img, 1)
    
    if canvas is None:
        canvas = np.zeros_like(img)
    
    # Cek apakah sedang di intro screen
    if show_intro:
        # Play audio intro (hanya sekali)
        play_intro_audio()
        
        # Tampilkan intro screen
        draw_attract_mode(img)
        cv2.imshow("DO THE MATH", img)
    else:
        # Mode normal (drawing)
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
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("Keluar dari program...")
        break
    elif key == ord(' '):
        # Toggle intro screen
        toggle_intro_mode()

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("Program selesai!")