"""
DO THE MATH! - Interactive Math Learning with Gesture Recognition
=================================================================

Program pembelajaran matematika interaktif yang menggabungkan:
1. Computer Vision & Gesture Recognition (Hand Tracking)
2. AI Digit Recognition (ONNX MNIST Model with Multi-Digit Support)
3. Quiz Management System
4. Audio Playback (Soal & Sound Effects)

Gesture yang tersedia:
- 1 Jari (Telunjuk): Menggambar
- 4 Jari (Tanpa Jempol): Submit & Check Answer
- 5 Jari: Hapus Canvas
- Q: Quit | R: Restart (di final screen)

"""

import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector

# Import modul digit recognition (file Anda dengan multi-digit support)
from digit_recognition import load_model, recognize_multi_digit

# Import modul quiz dan audio
from quiz_manager import QuizManager
from audio_player import AudioPlayer

# Import modul confetti effect
from confetti_effect import ConfettiSystem

# --- Konstanta ---
DRAW_CHARGE_TIME = 30  # Frame untuk aktivasi drawing
NOTIFICATION_DURATION = 60  # Durasi notifikasi (frame)
BRUSH_SIZE = 20  # Ukuran brush untuk menggambar

# Window size
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(3, WINDOW_WIDTH)
cap.set(4, WINDOW_HEIGHT)

# Inisialisasi Hand Detector
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, 
                        detectionCon=0.7, minTrackCon=0.5)

# --- INISIALISASI SISTEM ---
print("\n" + "="*60)
print("DO THE MATH! - INISIALISASI SISTEM")
print("="*60)

# Load ONNX MNIST model
print("\n[1/3] Loading ONNX Digit Recognition Model...")
try:
    session = load_model()
    RECOGNIZER_LOADED = True
except Exception as e:
    print(f"Error loading model: {e}")
    RECOGNIZER_LOADED = False
    session = None

# Initialize Quiz Manager
print("\n[2/3] Initializing Quiz Manager...")
try:
    quiz = QuizManager(data_folder="data", answers_file="answers.txt")
except Exception as e:
    print(f"\n✗ Error loading quiz: {str(e)}")
    print("Pastikan folder 'data/' dan file 'answers.txt' sudah tersedia!")
    exit(1)

# Initialize Audio Player
print("\n[3/3] Initializing Audio Player...")
audio = AudioPlayer(sounds_folder="sounds")
first_audio_played = False  

# Initialize Confetti System
print("\n[4/4] Initializing Confetti System...")
confetti = ConfettiSystem(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT)

print("\n" + "="*60)
print("SISTEM SIAP!")
print("="*60 + "\n")

# Initialize first question
current_question = quiz.get_current_question()

# Variabel global
previousPosition = None
canvas = None
draw_charge_counter = 0
is_drawing_allowed = False
notification_text = ""
notification_timer = 0
quiz_finished = False
pending_audio_path = None
answer_audio_playing = False  # Flag untuk track audio jawaban sedang playing


def getHandInfo(img):
    """
    Mendeteksi tangan dan mengembalikan informasi jari dan landmark.
    
    Args:
        img (numpy.ndarray): Frame gambar dari webcam
        
    Returns:
        tuple: (fingers, lmList) - Status jari dan list landmark
        None: Jika tidak ada tangan yang terdeteksi
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
    Menggambar di canvas berdasarkan gesture.
    
    Mode gesture:
    - 1 jari (telunjuk): Menggambar (dengan loading bar)
    - 4 jari (tanpa jempol): Submit & Check Answer
    - 5 jari: Hapus canvas
    
    Args:
        info (tuple): Informasi fingers dan landmark dari getHandInfo()
        previousPosition (tuple): Posisi sebelumnya untuk menggambar garis
        canvas (numpy.ndarray): Canvas untuk menggambar
        img (numpy.ndarray): Frame gambar dari webcam
        
    Returns:
        tuple: (currentPosition, canvas) - Posisi saat ini dan canvas yang diupdate
    """
    global draw_charge_counter, is_drawing_allowed, notification_text, notification_timer
    global quiz_finished
    
    # Block gesture jika audio jawaban sedang playing
    if answer_audio_playing:
        return previousPosition, canvas  # Return tanpa proses gesture

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
            cv2.line(canvas, currentPosition, previousPosition, (255, 255, 255), BRUSH_SIZE)
            
            # Gambar lingkaran kecil di posisi saat ini
            cv2.circle(canvas, currentPosition, 5, (255, 255, 255), cv2.FILLED)
            
            previousPosition = currentPosition
    
    # Mode hapus: semua jari terangkat
    elif fingers == [1, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)
        if notification_timer == 0:
            notification_text = "CANVAS DIHAPUS!"
            notification_timer = NOTIFICATION_DURATION // 2
            print("\n[INFO] Canvas dihapus!")
        
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    # Mode submit & prediksi: 4 jari tanpa jempol
    elif fingers == [0, 1, 1, 1, 1]:
        if notification_timer == 0 and not quiz_finished:
            process_answer(canvas, img)
            canvas = np.zeros_like(img)
        
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    # Mode idle: reset semua
    else:
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    return currentPosition, canvas


def process_answer(canvas, img):
    """
    Proses jawaban user: preprocess, predict, check, dan update quiz.
    Menggunakan multi-digit recognition untuk mendukung jawaban 2 digit.
    
    Flow audio:
    - Jawaban BENAR: Play correct sound -> applause (sequential)
    - Jawaban SALAH: Play wrong sound
    
    Args:
        canvas (numpy.ndarray): Canvas berisi gambar jawaban user
        img (numpy.ndarray): Frame webcam
    """
    global notification_text, notification_timer, quiz_finished, answer_audio_playing
    
    print("\n" + "="*60)
    print("MEMPROSES JAWABAN...")
    print("="*60)
    
    # Stop audio soal
    audio.stop_audio()
    
    if not RECOGNIZER_LOADED:
        print("⚠ Model tidak ter-load!")
        notification_text = "MODEL NOT LOADED!"
        notification_timer = NOTIFICATION_DURATION
        return
    
    # Prediksi digit menggunakan multi-digit recognition
    result, confidence = recognize_multi_digit(session, canvas, max_digits=3)
    
    if result is None or confidence < 50:
        print("⚠ Tidak dapat memproses gambar atau confidence terlalu rendah")
        notification_text = "GAMBAR TIDAK JELAS!"
        notification_timer = NOTIFICATION_DURATION
        return
    
    # Convert string result ke integer
    try:
        predicted_answer = int(result)
    except ValueError:
        print(f"⚠ Hasil prediksi tidak valid: {result}")
        notification_text = "PREDIKSI ERROR!"
        notification_timer = NOTIFICATION_DURATION
        return
    
    print(f"Hasil prediksi: {predicted_answer}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Check jawaban
    is_correct = quiz.check_answer(predicted_answer)
    
    current_question = quiz.get_current_question()
    correct_answer = current_question['correct_answer']
    
    # Set flag bahwa audio jawaban sedang playing
    answer_audio_playing = True
    
    if is_correct:
        # BENAR! Play sequential audio: correct -> applause
        print("✓ JAWABAN BENAR!")
        notification_text = f"BENAR! Jawaban: {correct_answer}"
        audio.play_correct_sequence()  # Sequential: correct -> applause
        confetti.generate_burst(num_particles=150)  # -> Trigger confetti effect
    else:
        # SALAH - Play wrong sound only
        print(f"✗ JAWABAN SALAH! Jawaban yang benar: {correct_answer}")
        notification_text = f"SALAH! Jawaban: {correct_answer}"
        audio.play_wrong_sound()  # Single audio: wrong
    
    notification_timer = NOTIFICATION_DURATION
    
    # Pindah ke soal berikutnya
    if not quiz.is_finished():
        next_question = quiz.next_question()
        if next_question:
            print(f"\n[INFO] Pindah ke {quiz.get_progress()}")
            # Simpan audio path untuk diputar nanti setelah sound effect selesai
            global pending_audio_path
            pending_audio_path = next_question['audio_path']
        else:
            # next_question adalah None, berarti quiz sudah selesai setelah increment
            quiz_finished = True
            answer_audio_playing = False
            print("\n" + "="*60)
            print("QUIZ SELESAI!")
            print("="*60)
    else:
        # Quiz selesai (edge case jika is_finished() sudah True sebelum increment)
        quiz_finished = True
        answer_audio_playing = False 
        print("\n" + "="*60)
        print("QUIZ SELESAI!")
        print("="*60)


def display_question_image(img, question):
    """
    Menampilkan gambar soal di KIRI ATAS window.
    
    Args:
        img (numpy.ndarray): Frame untuk overlay gambar soal
        question (dict): Question object dari quiz manager
    """
    if question is None:
        return
    
    try:
        # Load gambar soal
        question_img = cv2.imread(question['image_path'])
        
        if question_img is None:
            return
        
        # Resize gambar soal: dari 1920x1080 ke 380x214 (maintain aspect ratio 16:9)
        target_width = 380
        target_height = 214
        resized_question = cv2.resize(question_img, (target_width, target_height), 
                                     interpolation=cv2.INTER_AREA)
        
        # Posisi: KIRI ATAS (20, 20)
        x_offset = 20
        y_offset = 20
        
        # Overlay gambar soal ke frame
        img[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_question
        
        # Tambahkan border di sekitar gambar
        cv2.rectangle(img, (x_offset-2, y_offset-2), 
                     (x_offset+target_width+2, y_offset+target_height+2), 
                     (255, 255, 255), 3)
        
    except Exception as e:
        print(f"⚠ Error displaying question image: {str(e)}")


def displayInstructions(img):
    """
    Menampilkan instruksi di KANAN ATAS.
    
    Args:
        img (numpy.ndarray): Frame untuk overlay instruksi
    """
    # Background box untuk instruksi - KANAN ATAS
    box_x = WINDOW_WIDTH - 270
    box_y = 20
    box_w = 250
    box_h = 200
    
    # Background Box Medium Purple dalam Format BGR
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (219, 112, 147), -1) 

    # Border Box Deep Pink dalam Format BGR
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 20, 147), 3) 
    
    # Header
    cv2.putText(img, "INSTRUKSI:", (box_x + 10, box_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    instructions = [
        "1 Jari = Draw",
        "4 Jari = Submit",
        "5 Jari = Clear",
        "Q = Quit"
    ]
    
    y_offset = box_y + 60
    for i, text in enumerate(instructions):
        cv2.putText(img, text, (box_x + 15, y_offset + (i * 32)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def display_score_box(img):
    """
    Menampilkan skor di KIRI BAWAH (di bawah gambar soal).
    
    Args:
        img (numpy.ndarray): Frame untuk overlay skor
    """
    # Background box untuk skor - KIRI BAWAH
    box_x = 20
    box_y = 250  # Di bawah gambar soal (214 + 20 + margin)
    box_w = 380
    box_h = 140
    
    # Background Box Deep Pink Format BGR
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 20, 147), -1)

    # Border Box Gold Format BGR
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 215, 255), 3)
    
    # Header
    cv2.putText(img, "SKOR:", (box_x + 10, box_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Get score
    correct, total, percentage = quiz.get_score()
    progress = quiz.get_progress()
    
    # Tampilkan info
    info_lines = [
        f"{progress}",
        f"Benar: {correct} | Salah: {total - correct}",
        f"Persentase: {percentage:.0f}%"
    ]
    
    y_offset = box_y + 60
    for i, text in enumerate(info_lines):
        cv2.putText(img, text, (box_x + 15, y_offset + (i * 32)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def displayReadyButton(img):
    """
    Menampilkan status ready/drawing button di KANAN BAWAH (di bawah instruksi).
    
    Args:
        img (numpy.ndarray): Frame untuk menampilkan button
    """
    global draw_charge_counter, is_drawing_allowed
    
    # Posisi KANAN BAWAH - di bawah instruksi
    button_x = WINDOW_WIDTH - 270
    button_y = 240  # Di bawah box instruksi (220 + 20)
    button_w = 250
    button_h = 50
    
    if draw_charge_counter > 0 and not is_drawing_allowed:
        # Charging state
        progress = draw_charge_counter / DRAW_CHARGE_TIME
        
        # Background
        overlay = img.copy()
        cv2.rectangle(overlay, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Border
        cv2.rectangle(img, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (0, 255, 255), 3)
        
        # Progress bar
        fill_w = int(button_w * progress)
        cv2.rectangle(img, (button_x, button_y), (button_x + fill_w, button_y + button_h),
                     (0, 255, 0), -1)
        
        # Text
        cv2.putText(img, "SIAP...", (button_x + 70, button_y + 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    elif is_drawing_allowed:
        # Drawing state
        overlay = img.copy()
        cv2.rectangle(overlay, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (128, 0, 128), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Border
        cv2.rectangle(img, (button_x, button_y), (button_x + button_w, button_y + button_h),
                     (255, 0, 255), 3)
        
        # Text
        cv2.putText(img, "MENGGAMBAR...", (button_x + 20, button_y + 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def displayNotification(img):
    """
    Menampilkan notifikasi BENAR/SALAH di KIRI BAWAH (di bawah skor).
    
    Args:
        img (numpy.ndarray): Frame untuk menampilkan notifikasi
    """
    global notification_text, notification_timer
    
    if notification_timer > 0:
        # Posisi KIRI BAWAH - di bawah box skor
        box_x = 20
        box_y = 390  # Di bawah skor box (250 + 120 + 20)
        box_w = 380
        box_h = 80
        
        # Pilih warna berdasarkan notifikasi
        if "BENAR" in notification_text:
            bg_color = (0, 150, 0)  # Hijau gelap
            text_color = (0, 255, 0)  # Hijau terang
            border_color = (0, 255, 0)
        elif "SALAH" in notification_text:
            bg_color = (0, 0, 150)  # Merah gelap
            text_color = (0, 100, 255)  # Merah terang
            border_color = (0, 0, 255)
        elif "JELAS" in notification_text or "ERROR" in notification_text:
            bg_color = (0, 80, 150)  # Orange gelap
            text_color = (0, 165, 255)  # Orange terang
            border_color = (0, 165, 255)
        else:
            bg_color = (0, 100, 100)  # Kuning gelap
            text_color = (0, 255, 255)  # Kuning terang
            border_color = (0, 255, 255)
        
        # Background box dengan transparansi
        overlay = img.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h),
                     bg_color, -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # Border
        cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h),
                     border_color, 3)
        
        # Text - centered dalam box
        text_size = cv2.getTextSize(notification_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = box_x + (box_w - text_size[0]) // 2
        text_y = box_y + (box_h + text_size[1]) // 2
        
        cv2.putText(img, notification_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        notification_timer -= 1

def create_gradient_background(img, x1, y1, x2, y2, color_top, color_bottom):
    """
    Membuat gradient background vertikal pada region tertentu.
    
    Args:
        img (numpy.ndarray): Image untuk drawing
        x1, y1: Koordinat top-left corner
        x2, y2: Koordinat bottom-right corner
        color_top (tuple): Warna BGR untuk bagian atas
        color_bottom (tuple): Warna BGR untuk bagian bawah
    """
    height = y2 - y1
    
    # Loop untuk setiap baris dan buat gradient
    for i in range(height):
        # Hitung ratio untuk interpolasi warna (0.0 di atas, 1.0 di bawah)
        ratio = i / height
        
        # Interpolasi warna antara top dan bottom
        b = int(color_top[0] * (1 - ratio) + color_bottom[0] * ratio)
        g = int(color_top[1] * (1 - ratio) + color_bottom[1] * ratio)
        r = int(color_top[2] * (1 - ratio) + color_bottom[2] * ratio)
        
        # Draw horizontal line dengan warna yang sudah di-interpolasi
        cv2.line(img, (x1, y1 + i), (x2, y1 + i), (b, g, r), 1)

def show_final_score_screen(img):
    """
    Menampilkan layar skor akhir setelah quiz selesai.
    
    Args:
        img (numpy.ndarray): Frame untuk overlay final score
    """
    
    # Semi-transparent dark overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # Get final score
    correct, total, percentage = quiz.get_score()
    
    # Koordinat box
    box_x1 = 340
    box_y1 = 150
    box_x2 = 940
    box_y2 = 550

    # Gradient background (Medium Purple ke Dark Magenta)
    color_top = (219, 112, 147)      # BGR: Medium Purple
    color_bottom = (130, 0, 139)     # BGR: Dark Magenta/Indigo
    create_gradient_background(img, box_x1, box_y1, box_x2, box_y2, color_top, color_bottom)

    # Border tebal dengan warna Gold
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 215, 255), 5)
    
    # Title
    cv2.putText(img, "QUIZ SELESAI!", (430, 230), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 215, 255), 3)
    
    # Score info
    cv2.putText(img, f"Skor Anda: {correct}/{total}", (450, 310), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.putText(img, f"Persentase: {percentage:.1f}%", (450, 370), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    # Rating berdasarkan persentase
    if percentage >= 90:
        rating = "EXCELLENT!"
        color = (0, 255, 0) # Hijau
    elif percentage >= 75:
        rating = "GREAT!"
        color = (0, 255, 255) # Kuning
    elif percentage >= 60:
        rating = "GOOD!"
        color = (255, 255, 0) # Cyan
    else:
        rating = "KEEP TRYING!"
        color = (0, 165, 255) # Orange
    
    cv2.putText(img, rating, (500, 430), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Instructions
    cv2.putText(img, "Tekan 'R' untuk Restart", (450, 490), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 200), 2)
    cv2.putText(img, "Tekan 'Q' untuk Keluar", (450, 525), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 200), 2)


# Main loop
print("Program dimulai! Tekan 'Q' untuk keluar.\n")
print("=" * 60)
print("CARA PENGGUNAAN:")
print("- Angkat 1 jari (telunjuk) untuk menggambar")
print("- Angkat 4 jari (tanpa jempol) untuk submit jawaban")
print("- Angkat 5 jari untuk menghapus canvas")
print("=" * 60 + "\n")

while True:
    success, img = cap.read()
    
    if not success:
        print("Gagal membaca frame dari webcam")
        break
    
    img = cv2.flip(img, 1)

    # Play audio soal pertama dengan delay setelah window muncul
    if not first_audio_played:
        cv2.imshow("DO THE MATH! - Interactive Math Learning", img)
        cv2.waitKey(500)  # Delay 0.5 detik
        if current_question:
            audio.play_question_audio(current_question['audio_path'])
        first_audio_played = True
    
    if canvas is None:
        canvas = np.zeros_like(img)
    
    # Deteksi tangan
    info = getHandInfo(img)
    fingers = None
    
    # Proses gesture hanya jika tidak sedang playing audio jawaban
    if info and not quiz_finished and not answer_audio_playing:
        fingers, lmlist = info
        previousPosition, canvas = draw(info, previousPosition, canvas, img)
    else:
        previousPosition = None
        is_drawing_allowed = False
        draw_charge_counter = 0
    
    # Gabungkan image dengan canvas
    combinedImage = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    
    # Update audio sequence (untuk sequential playback)
    audio.update_sequence()

    # Update confetti system  
    if confetti.is_active():  
        confetti.update()  
    
    # Check apakah audio jawaban sudah selesai
    if answer_audio_playing:
        # Jika audio sequence sudah selesai (untuk correct) atau audio single sudah selesai (untuk wrong)
        if not audio.is_sequence_playing() and not audio.is_playing():
            # Audio jawaban selesai, set flag ke False
            answer_audio_playing = False
            
            # Clear canvas setelah audio selesai (jika belum di-clear)
            if np.any(canvas):
                canvas = np.zeros_like(img)
            
            # Play audio soal berikutnya jika ada pending
            if pending_audio_path and not quiz_finished:
                audio.play_question_audio(pending_audio_path)
                pending_audio_path = None
    
    if not quiz_finished:
        # Tampilkan UI normal
        current_question = quiz.get_current_question()
        
        # KIRI ATAS: Gambar Soal
        display_question_image(combinedImage, current_question)
        
        # KIRI BAWAH: Skor (di bawah gambar soal)
        display_score_box(combinedImage)
        
        # KIRI BAWAH: Notifikasi BENAR/SALAH (di bawah skor)
        displayNotification(combinedImage)
        
        # KANAN ATAS: Instruksi
        displayInstructions(combinedImage)
        
        # KANAN BAWAH: Status Drawing (di bawah instruksi)
        if fingers is not None:
            displayReadyButton(combinedImage)
        
        # Render confetti effect (di atas semua UI)
        if confetti.is_active():
            confetti.draw(combinedImage)
    else:
        # Tampilkan final score screen
        show_final_score_screen(combinedImage)
    
    cv2.imshow("DO THE MATH! - Interactive Math Learning", combinedImage)
    
    # Keyboard control
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n" + "="*60)
        print("Program dihentikan.")
        print("="*60)
        break
    elif key == ord('r') and quiz_finished:
        # Restart quiz
        quiz.reset()
        quiz_finished = False
        canvas = np.zeros_like(img)
        notification_text = ""
        notification_timer = 0
        confetti.clear()
        
        current_question = quiz.get_current_question()
        if current_question:
            audio.play_question_audio(current_question['audio_path'])

# Cleanup
audio.cleanup()
cap.release()
cv2.destroyAllWindows()