import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

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
    Menggambar di canvas berdasarkan gesture
    - 1 jari (telunjuk): Menggambar
    - 5 jari: Hapus canvas
    - 4 jari (tanpa jempol): Mode siap/tidak menggambar
    """
    fingers, lmlist = info
    currentPosition = None
    
    # Mode menggambar: hanya jari telunjuk yang terangkat
    if fingers == [0, 1, 0, 0, 0]:
        currentPosition = lmlist[8][0:2]  # Posisi ujung jari telunjuk
        
        if previousPosition is None:
            previousPosition = currentPosition
        
        # Gambar garis dari posisi sebelumnya ke posisi sekarang
        cv2.line(canvas, currentPosition, previousPosition, (255, 255, 255), 14)
        
        # Gambar lingkaran kecil di posisi saat ini
        cv2.circle(canvas, currentPosition, 5, (255, 255, 255), cv2.FILLED)
    
    # Mode hapus: semua jari terangkat
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)
        previousPosition = None
        print("Canvas dihapus!")
    
    # Mode siap kirim: 4 jari tanpa jempol
    elif fingers == [0, 1, 1, 1, 1]:
        print("Mode KIRIM - Siap mengirim gambar!")
        # Tambahkan teks di layar
        cv2.putText(canvas, "READY TO SEND", (750, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 250, 0), 3)
    
    return currentPosition, canvas

def displayInstructions(img):
    """
    Menampilkan instruksi penggunaan di layar
    """
    instructions = [
        "INSTRUKSI:",
        "1 Jari (Telunjuk) = Menggambar",
        "5 Jari = Hapus Canvas",
        "4 Jari (tanpa Jempol) = Mengirim Jawaban",
        "Tekan 'q' = Keluar"
    ]
    
    y_offset = 30
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (10, y_offset + (i * 30)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Main loop
print("Program dimulai! Tekan 'q' untuk keluar.")
print("\nCara penggunaan:")
print("- Angkat 1 jari (telunjuk) untuk menggambar")
print("- Angkat 5 jari untuk menghapus canvas")
print("- Angkat 4 jari (tanpa jempol) untuk mode kirim")

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
    
    if info:
        fingers, lmlist = info
        previousPosition, canvas = draw(info, previousPosition, canvas, img)
        
        # Tampilkan status jari di layar
        finger_status = f"Jari: {fingers}"
        cv2.putText(img, finger_status, (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Gabungkan gambar kamera dengan canvas
    combinedImage = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    
    # Tampilkan instruksi
    displayInstructions(combinedImage)
    
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