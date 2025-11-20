import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
from digit_recognition import predict

# --- Konstanta ---
DRAW_CHARGE_TIME = 20  # Frame untuk aktivasi drawing

# Drawing box (centered square for better prediction)
BOX_SIZE = 400  # 400x400 pixel box
BOX_X = (1280 - BOX_SIZE) // 2  # Centered horizontally
BOX_Y = (720 - BOX_SIZE) // 2   # Centered vertically
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
last_prediction = None
prediction_confidence = 0.0
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
            
            # Check if position is within drawing box
            if (BOX_X <= currentPosition[0] <= BOX_X + BOX_SIZE and 
                BOX_Y <= currentPosition[1] <= BOX_Y + BOX_SIZE):
                
                # Gambar garis dari posisi sebelumnya ke posisi sekarang
                cv2.line(canvas, currentPosition, previousPosition, (255, 255, 255), 14)
                
                # Gambar lingkaran kecil di posisi saat ini
                cv2.circle(canvas, currentPosition, 5, (255, 255, 255), cv2.FILLED)
                
                previousPosition = currentPosition
            else:
                # Outside box - just update position without drawing
                previousPosition = currentPosition
    
    # Mode hapus: semua jari terangkat
    elif fingers == [1, 1, 1, 1, 1]:
        if notification_timer == 0:
            canvas = np.zeros_like(img)
            notification_text = "CANVAS DIHAPUS!"
            notification_timer = NOTIFICATION_DURATION
            last_prediction = None
            prediction_confidence = 0.0
            print("Canvas dihapus!")
        
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    # Mode simpan: 4 jari tanpa jempol
    elif fingers == [0, 1, 1, 1, 1]:
        if notification_timer == 0:
            # Crop hanya area drawing box
            cropped_canvas = canvas[BOX_Y:BOX_Y+BOX_SIZE, BOX_X:BOX_X+BOX_SIZE]
            
            # Simpan gambar yang sudah di-crop
            cv2.imwrite("hasil_gambar.png", cropped_canvas)
            
            # Prediksi otomatis setelah menyimpan
            try:
                digit, probs = predict("hasil_gambar.png")
                last_prediction = digit
                prediction_confidence = probs[digit]
                notification_text = f"TERSIMPAN! Prediksi: {digit} ({probs[digit]*100:.1f}%)"
                print(f"Gambar disimpan. Prediksi: {digit} (Confidence: {probs[digit]*100:.2f}%)")
            except Exception as e:
                notification_text = "TERSIMPAN! (Gagal prediksi)"
                print(f"Gambar disimpan, tapi gagal prediksi: {e}")
            
            notification_timer = NOTIFICATION_DURATION
        
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

def displayDrawingBox(img):
    """
    Menampilkan kotak area menggambar
    """
    # Gambar border kotak dengan warna cyan
    cv2.rectangle(img, (BOX_X, BOX_Y), (BOX_X + BOX_SIZE, BOX_Y + BOX_SIZE), 
                 (255, 255, 0), 3)  # Cyan color, 3px thick
    
    # Label di atas kotak
    cv2.putText(img, "GAMBAR DI SINI", (BOX_X + 100, BOX_Y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

def displayPrediction(img):
    """
    Menampilkan hasil prediksi di layar
    """
    global last_prediction, prediction_confidence
    
    if last_prediction is not None:
        # Background box untuk prediksi
        box_x, box_y, box_w, box_h = 1000, 20, 260, 180
        
        # Buat overlay semi-transparan
        overlay = img.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                     (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Border
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                     (0, 255, 0), 3)
        
        # Judul
        cv2.putText(img, "PREDIKSI:", (box_x + 20, box_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Angka prediksi (besar)
        cv2.putText(img, str(last_prediction), (box_x + 100, box_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        
        # Confidence
        conf_text = f"Confidence: {prediction_confidence*100:.1f}%"
        cv2.putText(img, conf_text, (box_x + 20, box_y + 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
    
    # Tampilkan kotak area menggambar
    displayDrawingBox(combinedImage)
    
    # Tampilkan instruksi
    displayInstructions(combinedImage)
    
    # Tampilkan loading bar jika sedang charging
    if fingers is not None:
        displayLoadingBar(combinedImage, fingers)
    
    # Tampilkan notifikasi
    displayNotification(combinedImage)
    
    # Tampilkan prediksi
    displayPrediction(combinedImage)
    
    # Tampilkan hasil
    cv2.imshow("Gesture Drawing - Combined View", combinedImage)
    # cv2.imshow("Canvas Only", canvas)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Program dihentikan.")
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()