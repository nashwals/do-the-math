import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# --- Variabel Constant ---
DRAW_CHARGE_TIME = 30 
NOTIFICATION_DURATION = 30

# Variabel target resolusi
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

def get_hand_info(img, detector):
    """
    Mendeteksi tangan dalam gambar dan mengembalikan informasi gestur.

    Args:
        img (numpy.ndarray): Frame gambar dari webcam.
        detector (HandDetector): Objek HandDetector dari cvzone.

    Returns:
        tuple: (fingers, lmList) jika tangan terdeteksi, 
               None jika tidak ada tangan.
    """
    hands, img_with_hands = detector.findHands(img, draw=False, flipType=True)

    if hands:
        hand1 = hands[0]
        lmList = hand1["lmList"]      
        fingers = detector.fingersUp(hand1) 
        return fingers, lmList
    
    return None

def main():
    """
    Fungsi utama untuk menjalankan aplikasi deteksi gestur dan kanvas.
    Fungsi ini menangani inisialisasi, loop utama, dan tampilan.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka webcam.")
        return
        
    cap.set(3, TARGET_WIDTH) 
    cap.set(4, TARGET_HEIGHT)  

    detector = HandDetector(detectionCon=0.7, maxHands=1)

    # --- Variabel Status (State) ---
    prev_pos = None  
    canvas = None    
    
    notification_text = ""
    notification_timer = 0
    
    draw_charge_counter = 0  
    is_drawing_allowed = False 

    print("Aplikasi berjalan... Tekan 'q' untuk keluar.")
    print("Gestur: Tahan 1 Jari (Gambar), 5 Jari (Hapus), 4 Jari (Kirim)")

    # Loop utama program
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Gagal membaca frame.")
            break
        
        img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))

        img = cv2.flip(img, 1) 

        if canvas is None:
            canvas = np.zeros_like(img)

        # Inisialisasi 'fingers' 
        fingers = None

        # 3. Deteksi Tangan
        info = get_hand_info(img, detector)

        # 4. Proses Logika Gestur dan Status
        if info:
            fingers, lmlist = info
            current_pos = lmlist[8][0:2] # Ambil posisi jari telunjuk

            # --- Aksi 1: Tulis (Hanya Jari Telunjuk) ---
            if fingers == [0, 1, 0, 0, 0]:
                if not is_drawing_allowed:
                    draw_charge_counter += 1
                    if draw_charge_counter >= DRAW_CHARGE_TIME:
                        is_drawing_allowed = True
                        prev_pos = current_pos 
                
                if is_drawing_allowed:
                    if prev_pos is None: prev_pos = current_pos
                    cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)
                    prev_pos = current_pos
                
            # --- Aksi 2: Hapus (Semua 5 Jari) ---
            elif fingers == [1, 1, 1, 1, 1]:
                if notification_timer == 0:
                    canvas = np.zeros_like(img) 
                    notification_text = "KANVAS DIHAPUS"
                    notification_timer = NOTIFICATION_DURATION
                
                prev_pos = None
                is_drawing_allowed = False
                draw_charge_counter = 0

            # --- Aksi 3: Kirim (4 Jari, tanpa Jempol) ---
            elif fingers == [0, 1, 1, 1, 1]:
                if notification_timer == 0:
                    print("Gestur 'Kirim' terdeteksi! Menyimpan gambar...")
                    cv2.imwrite("hasil_gambar.png", canvas)
                    notification_text = "TERKIRIM!"
                    notification_timer = NOTIFICATION_DURATION

                prev_pos = None
                is_drawing_allowed = False
                draw_charge_counter = 0

            # --- Kondisi Lain (Idle) ---
            else:
                prev_pos = None
                is_drawing_allowed = False
                draw_charge_counter = 0

        else:
            # Jika tidak ada tangan terdeteksi
            prev_pos = None
            is_drawing_allowed = False
            draw_charge_counter = 0

        # 5. Gabungkan gambar webcam dan kanvas
        combined_img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

        # 6. Tampilkan UI / Notifikasi
        
        if notification_timer > 0:
            cv2.putText(combined_img, notification_text, (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            notification_timer -= 1
        
        # Cek 'fingers' untuk gestur menggambar
        elif fingers == [0, 1, 0, 0, 0]: 
            if is_drawing_allowed:
                cv2.putText(combined_img, "MENGGAMBAR...", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                cv2.putText(combined_img, "SIAP...", (50, 125), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Gambar 'Loading Bar'
                cv2.rectangle(combined_img, (50, 50), (250, 80), (0, 0, 255), 3)
                bar_width = int((draw_charge_counter / DRAW_CHARGE_TIME) * 200)
                cv2.rectangle(combined_img, (50, 50), (50 + bar_width, 80), 
                              (0, 255, 0), cv2.FILLED)

        # 7. Tampilkan Hasil (Window Utama)
        cv2.imshow("Gesture Canvas (Tekan 'q' untuk keluar)", combined_img)
        
        # 8. Tombol Keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Menutup aplikasi...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()