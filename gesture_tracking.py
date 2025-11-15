import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

# --- Konstanta ---
DRAW_CHARGE_TIME = 30  # Frame untuk aktivasi drawing
NOTIFICATION_DURATION = 30  # Durasi notifikasi (frame)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Inisialisasi Hand Detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, 
                        detectionCon=0.7, minTrackCon=0.5)

# Variabel untuk menyimpan posisi sebelumnya dan canvas
previousPosition = None
canvas = None

# Variabel untuk fitur baru
draw_charge_counter = 0
is_drawing_allowed = False
notification_text = ""
notification_timer = 0

def getHandInfo(img):
    """
    Mendeteksi tangan dan mengembalikan informasi jari dan landmark
    """
    hands, img = detector.findHands(img, draw=True, flipType=True)
    
    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]  # List of 21 landmarks
        fingers = detector.fingersUp(hand1)  # Status jari (0=lipat, 1=tegak)
        return fingers, lmList
    else:
        return None

def draw(info, previousPosition, canvas, img):
    """
    Menggambar di canvas berdasarkan gesture dengan loading bar
    - 1 jari (telunjuk): Menggambar (dengan loading bar)
    - 5 jari: Hapus canvas
    - 4 jari (tanpa jempol): Simpan gambar
    """
    global draw_charge_counter, is_drawing_allowed, notification_text, notification_timer
    
    fingers, lmlist = info
    currentPosition = None
    
    # Mode menggambar: hanya jari telunjuk yang terangkat
    if fingers == [0, 1, 0, 0, 0]:
        currentPosition = lmlist[8][0:2]  # Posisi ujung jari telunjuk
        
        # Logika charging untuk aktivasi drawing
        if not is_drawing_allowed:
            draw_charge_counter += 1
            if draw_charge_counter >= DRAW_CHARGE_TIME:
                is_drawing_allowed = True
                previousPosition = currentPosition
        
        # Mulai menggambar setelah charging selesai
        if is_drawing_allowed:
            if previousPosition is None:
                previousPosition = currentPosition
            
            # Gambar garis dari posisi sebelumnya ke posisi sekarang
            cv2.line(canvas, currentPosition, previousPosition, (255, 255, 255), 14)
            
            # Gambar lingkaran kecil di posisi saat ini
            cv2.circle(canvas, currentPosition, 5, (255, 255, 255), cv2.FILLED)
            
            previousPosition = currentPosition
    
    # Mode hapus: semua jari terangkat
    elif fingers == [1, 1, 1, 1, 1]:
        if notification_timer == 0:
            canvas = np.zeros_like(img)
            notification_text = "CANVAS DIHAPUS!"
            notification_timer = NOTIFICATION_DURATION
            print("Canvas dihapus!")
        
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    # Mode simpan: 4 jari tanpa jempol
    elif fingers == [0, 1, 1, 1, 1]:
        if notification_timer == 0:
            cv2.imwrite("hasil_gambar.png", canvas)
            notification_text = "GAMBAR TERSIMPAN!"
            notification_timer = NOTIFICATION_DURATION
            print("Gambar disimpan sebagai 'hasil_gambar.png'")
        
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    # Mode idle: reset semua
    else:
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    return currentPosition, canvas

def displayInstructions(img):
    """
    Menampilkan instruksi penggunaan di layar
    """
    instructions = [
        "INSTRUKSI:",
        "1 Jari (Telunjuk) = Menggambar",
        "5 Jari = Hapus Canvas",
        "4 Jari (tanpa Jempol) = Simpan Gambar",
        "Tekan 'q' = Keluar"
    ]
    
    y_offset = 30
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, y_offset + (i * 30)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def displayLoadingBar(img, fingers):
    """
    Menampilkan loading bar saat charging mode drawing
    """
    global draw_charge_counter, is_drawing_allowed
    
    # Tampilkan loading bar hanya saat gesture menggambar dan belum aktif
    if fingers == [0, 1, 0, 0, 0]:
        if is_drawing_allowed:
            # Tampilkan status "MENGGAMBAR"
            cv2.putText(img, "MENGGAMBAR...", (50, 230), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        else:
            # Tampilkan "SIAP..." dan loading bar
            cv2.putText(img, "SIAP...", (50, 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Gambar border loading bar
            cv2.rectangle(img, (50, 230), (250, 260), (0, 0, 255), 3)
            
            # Gambar progress loading bar
            bar_width = int((draw_charge_counter / DRAW_CHARGE_TIME) * 200)
            cv2.rectangle(img, (50, 230), (50 + bar_width, 260), 
                         (0, 255, 0), cv2.FILLED)

def displayNotification(img):
    """
    Menampilkan notifikasi sementara di layar
    """
    global notification_text, notification_timer
    
    if notification_timer > 0:
        cv2.putText(img, notification_text, (400, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        notification_timer -= 1

# Main loop
print("Program dimulai! Tekan 'q' untuk keluar.")
print("\nCara penggunaan:")
print("- Angkat 1 jari (telunjuk) untuk menggambar (tahan hingga loading selesai)")
print("- Angkat 5 jari untuk menghapus canvas")
print("- Angkat 4 jari (tanpa jempol) untuk menyimpan gambar")

while True:
    success, img = cap.read()
    
    if not success:
        print("Gagal membaca frame dari webcam")
        break
    
    # Flip gambar agar seperti cermin
    img = cv2.flip(img, 1)
    
    # Inisialisasi canvas jika belum ada
    if canvas is None:
        canvas = np.zeros_like(img)
    
    # Deteksi tangan dan dapatkan informasi
    info = getHandInfo(img)
    
    fingers = None  # Inisialisasi untuk cek di luar blok if
    
    if info:
        fingers, lmlist = info
        previousPosition, canvas = draw(info, previousPosition, canvas, img)
        
        # Tampilkan status jari di layar
        finger_status = f"Jari: {fingers}"
        cv2.putText(img, finger_status, (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        # Reset state jika tidak ada tangan
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    # Gabungkan gambar kamera dengan canvas
    combinedImage = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    
    # Tampilkan instruksi
    displayInstructions(combinedImage)
    
    # Tampilkan loading bar jika sedang charging
    if fingers is not None:
        displayLoadingBar(combinedImage, fingers)
    
    # Tampilkan notifikasi
    displayNotification(combinedImage)
    
    # Tampilkan hasil
    cv2.imshow("Gesture Drawing - Combined View", combinedImage)
    cv2.imshow("Canvas Only", canvas)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Program dihentikan.")
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()