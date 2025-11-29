"""
DO THE MATH! - Interactive Math Learning with Gesture Recognition
=================================================================

Main Program untuk pembelajaran matematika interaktif yang menggabungkan:
1. Computer Vision & Gesture Recognition (Hand Tracking)
2. AI Digit Recognition (ONNX MNIST Model with Multi-Digit Support)
3. Quiz Management System
4. Audio Playback (Soal & Sound Effects)
5. Confetti Particle System

Gesture yang tersedia:
- 1 Jari (Telunjuk): Menggambar
- 4 Jari (Tanpa Jempol): Submit & Check Answer
- 5 Jari: Hapus Canvas
- Q: Quit | R: Restart (di final screen)
"""

import numpy as np
import cv2
import math
import time
import os
from typing import Optional

# Import modul-modul yang diperlukan
from gesture_tracking import GestureController
from digit_recognition import load_model, recognize_multi_digit
from quiz_manager import QuizManager
from audio_manager import AudioPlayer
from confetti_effect import ConfettiSystem


# --- KONSTANTA ---
DRAW_CHARGE_TIME = 30  # Frame untuk aktivasi drawing
NOTIFICATION_DURATION = 60  # Durasi notifikasi (frame)
BRUSH_SIZE = 20  # Ukuran brush untuk menggambar

# Window size
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Drawing area constants
DRAWING_AREA_X = 300  # Lebar panel kiri untuk instruksi


# ======================= UI MANAGER =======================
class UIManager:
    """
    Class untuk mengelola semua UI rendering.
    
    Attributes:
        notification_text (str): Text notifikasi yang ditampilkan
        notification_timer (int): Timer countdown untuk notifikasi
        notification_color (tuple): Warna BGR untuk notifikasi
    """
    
    def __init__(self):
        """Inisialisasi UI Manager"""
        self.notification_text = ""
        self.notification_timer = 0
        self.notification_color = (255, 255, 255)
        
    def draw_left_panel(self, img, quiz_manager, gesture_controller):
        """
        Menggambar panel kiri dengan instruksi dan status
        
        Args:
            img (numpy.ndarray): Frame untuk drawing
            quiz_manager: QuizManager instance
            gesture_controller: GestureController instance
        """
        panel_width = DRAWING_AREA_X
        
        # Background panel kiri dengan transparansi
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, WINDOW_HEIGHT),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        
        # Border panel
        cv2.line(img, (panel_width, 0), (panel_width, WINDOW_HEIGHT),
                (100, 100, 100), 2)
        
        y_offset = 30
        
        # Title
        cv2.putText(img, "DO THE MATH!", (20, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 50
        
        # Progress
        total_questions = quiz_manager.get_total_questions()
        current_index = min(quiz_manager.current_index + 1, total_questions)
        progress_text = f"Soal: {current_index}/{total_questions}"
        cv2.putText(img, progress_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        # Score
        correct, _, _ = quiz_manager.get_score()
        score_text = f"Skor: {correct}"
        cv2.putText(img, score_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 50
        
        # Instruksi
        cv2.putText(img, "INSTRUKSI:", (20, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 35
        
        instructions = [
            "1 Jari",
            "  (telunjuk)",
            "  = Gambar jawaban",
            "",
            "4 Jari",
            "  (tanpa jempol)",
            "  = Submit jawaban",
            "",
            "5 Jari",
            "  = Hapus canvas",
            "",
            "Tekan 'Q' = Quit",
            "Tekan 'R' = Restart"
        ]
        
        for instruction in instructions:
            cv2.putText(img, instruction, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 25
        
        # Drawing status indicator
        y_offset += 20
        self._draw_status_indicator(img, gesture_controller, 20, y_offset)
    
    def _draw_status_indicator(self, img, gesture_controller, x_pos, y_pos):
        """
        Menggambar status drawing indicator
        
        Args:
            img (numpy.ndarray): Frame untuk drawing
            gesture_controller: GestureController instance
            x_pos (int): Posisi X
            y_pos (int): Posisi Y
        """
        # Status text
        if gesture_controller.is_ready_to_draw():
            status = "READY!"
            color = (0, 255, 0)
        else:
            status = "CHARGING..."
            color = (0, 165, 255)
        
        cv2.putText(img, status, (x_pos, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Progress bar
        progress = gesture_controller.get_draw_progress()
        bar_x = x_pos
        bar_y = y_pos + 15
        bar_w = 260
        bar_h = 20
        
        # Bar background
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (100, 100, 100), -1)
        
        # Bar fill
        fill_w = int(bar_w * progress)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                     (0, 255, 0), -1)
        
        # Bar border
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (255, 255, 255), 2)
    
    def draw_question_panel(self, img, quiz_manager):
        """
        Menggambar panel soal di bagian atas kanan
        
        Args:
            img (numpy.ndarray): Frame untuk drawing
            quiz_manager: QuizManager instance
        """
        question = quiz_manager.get_current_question()
        
        if question is None:
            return
        
        try:
            # Load gambar soal
            question_img = cv2.imread(question['image_path'])
            
            if question_img is None:
                return
            
            # Resize soal agar tidak terlalu besar
            max_width = 400
            max_height = 150
            
            h, w = question_img.shape[:2]
            scale = min(max_width / w, max_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized_question = cv2.resize(question_img, (new_w, new_h))
            
            # Posisi soal di kanan atas
            x_pos = WINDOW_WIDTH - new_w - 20
            y_pos = 20
            
            # Background untuk soal
            overlay = img.copy()
            padding = 10
            cv2.rectangle(overlay,
                         (x_pos - padding, y_pos - padding),
                         (x_pos + new_w + padding, y_pos + new_h + padding),
                         (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
            
            # Border hijau
            cv2.rectangle(img,
                         (x_pos - padding, y_pos - padding),
                         (x_pos + new_w + padding, y_pos + new_h + padding),
                         (0, 255, 0), 3)
            
            # Paste soal
            img[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = resized_question
            
        except Exception as e:
            print(f"⚠ Error displaying question: {str(e)}")
    
    def show_notification(self, text, color=(255, 255, 255), duration=None):
        """
        Menampilkan notifikasi
        
        Args:
            text (str): Text notifikasi
            color (tuple): Warna BGR notifikasi
            duration (int): Durasi notifikasi dalam frames
        """
        if duration is None:
            duration = NOTIFICATION_DURATION
            
        self.notification_text = text
        self.notification_color = color
        self.notification_timer = duration
    
    def draw_notification(self, img):
        """
        Menggambar notifikasi di layar
        
        Args:
            img (numpy.ndarray): Frame untuk drawing
        """
        if self.notification_timer > 0:
            # Posisi notifikasi di tengah bawah
            text_size = cv2.getTextSize(self.notification_text,
                                       cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
            
            text_x = (WINDOW_WIDTH - text_size[0]) // 2
            text_y = WINDOW_HEIGHT - 60
            
            # Background notifikasi
            overlay = img.copy()
            padding = 20
            cv2.rectangle(overlay,
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
            
            # Border
            cv2.rectangle(img,
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         self.notification_color, 2)
            
            # Text
            cv2.putText(img, self.notification_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, self.notification_color, 3)
            
            self.notification_timer -= 1

class MathQuizApp:
    """
    Main application class untuk DO THE MATH!
    
    Attributes:
        cap (cv2.VideoCapture): Webcam capture object
        gesture_controller (GestureController): Controller untuk gesture tracking
        quiz (QuizManager): Manager untuk quiz
        audio (AudioPlayer): Player untuk audio
        confetti (ConfettiSystem): System untuk confetti effect
        session: ONNX session untuk digit recognition
        canvas (numpy.ndarray): Canvas untuk drawing
        notification_text (str): Text notifikasi yang ditampilkan
        notification_timer (int): Timer untuk notifikasi
        quiz_finished (bool): Flag apakah quiz sudah selesai
        answer_audio_playing (bool): Flag audio jawaban sedang playing
        pending_audio_path (str): Path audio yang pending
        first_audio_played (bool): Flag audio pertama sudah diplay
        recognizer_loaded (bool): Flag apakah recognizer ter-load
    """
    
    def __init__(self):
        """
        Inisialisasi aplikasi dan semua komponennya.
        """
        print("\n" + "="*60)
        print("DO THE MATH! - INISIALISASI SISTEM")
        print("="*60)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, WINDOW_WIDTH)
        self.cap.set(4, WINDOW_HEIGHT)
        
        # Initialize gesture controller
        print("\n[1/4] Initializing Gesture Controller...")
        self.gesture_controller = GestureController(
            draw_charge_time=DRAW_CHARGE_TIME,
            brush_size=BRUSH_SIZE
        )
        
        # Load ONNX MNIST model
        print("\n[2/4] Loading ONNX Digit Recognition Model...")
        try:
            self.session = load_model()
            self.recognizer_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.recognizer_loaded = False
            self.session = None
        
        # Initialize Quiz Manager
        print("\n[3/4] Initializing Quiz Manager...")
        try:
            self.quiz = QuizManager(data_folder="data", answers_file="answers.txt")
        except Exception as e:
            print(f"\n✗ Error loading quiz: {str(e)}")
            print("Pastikan folder 'data/' dan file 'answers.txt' sudah tersedia!")
            exit(1)
        
        # Initialize Audio Player
        print("\n[4/4] Initializing Audio Player...")
        self.audio = AudioPlayer(sounds_folder="assets/audio/")

        # Load intro audio
        self.intro_audio_path = "assets/audio/opening.wav"
        self.intro_audio_loaded = False

        # Coba load intro audio dengan pygame.mixer.music
        try:
            if os.path.exists(self.intro_audio_path):
                import pygame
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load(self.intro_audio_path)
                self.intro_audio_loaded = True
                print(f"✓ Intro audio loaded: {self.intro_audio_path}")
            else:
                print(f"⚠ Intro audio not found: {self.intro_audio_path}")
        except Exception as e:
            print(f"⚠ Error loading intro audio: {str(e)}")
            self.intro_audio_loaded = False
        
        # Initialize Confetti System
        print("\n[5/5] Initializing Confetti System...")
        self.confetti = ConfettiSystem(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT)
        
        print("\n" + "="*60)
        print("SISTEM SIAP!")
        print("="*60 + "\n")
        
        # Initialize state variables
        self.ui_manager = UIManager()
        self.canvas: Optional[np.ndarray] = None
        self.quiz_finished = False
        self.answer_audio_playing = False
        self.pending_audio_path: Optional[str] = None
        self.first_audio_played = False
        
        # Intro screen state
        self.show_intro = True
        self.intro_audio_played = False
    
    def _get_scaled_font(self, base_size: float) -> float:
        """
        Mengambil ukuran font yang sudah di-scale berdasarkan window size.
        
        Args:
            base_size (float): Ukuran font dasar
            
        Returns:
            float: Ukuran font yang sudah di-scale
        """
        min_scale = min(WINDOW_WIDTH / 1920, WINDOW_HEIGHT / 1080)
        return max(0.4, base_size * min_scale)
    
    def _get_scaled_thickness(self, base_thickness: int) -> int:
        """
        Mengambil ketebalan garis yang sudah di-scale berdasarkan window size.
        
        Args:
            base_thickness (int): Ketebalan dasar
            
        Returns:
            int: Ketebalan yang sudah di-scale
        """
        min_scale = min(WINDOW_WIDTH / 1920, WINDOW_HEIGHT / 1080)
        return max(1, int(base_thickness * min_scale))
    
    def play_intro_audio(self):
        """
        Helper method untuk play audio intro.
        
        Menggunakan pygame.mixer.music untuk play background music intro.
        Audio hanya diputar sekali saat pertama kali masuk intro screen.
        """
        if self.intro_audio_loaded and not self.intro_audio_played:
            try:
                import pygame
                pygame.mixer.music.play()
                self.intro_audio_played = True
                print("♪ Playing intro audio...")
            except Exception as e:
                print(f"⚠ Error playing intro audio: {str(e)}")
                self.intro_audio_played = True  # Set True agar tidak retry terus
    
    def draw_intro_screen(self, img: np.ndarray):
        """
        Menampilkan intro screen / attract mode sebelum quiz dimulai.
        
        Intro screen menampilkan:
        - Judul game "DO THE MATH!"
        - Subtitle dan tagline
        - Instruksi cara bermain dengan icon
        - Call to action untuk mulai (tekan SPACE)
        
        Args:
            img (numpy.ndarray): Frame untuk display
        """
        # Buat overlay gelap di atas video
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Buat background pattern grid yang halus untuk texture
        min_scale = min(WINDOW_WIDTH / 1920, WINDOW_HEIGHT / 1080)
        pattern_overlay = img.copy()
        grid_spacing = int(80 * min_scale)
        
        # Gambar garis grid vertikal
        for x in range(0, WINDOW_WIDTH, grid_spacing):
            cv2.line(pattern_overlay, (x, 0), (x, WINDOW_HEIGHT), (30, 30, 30), 1)
        
        # Gambar garis grid horizontal
        for y in range(0, WINDOW_HEIGHT, grid_spacing):
            cv2.line(pattern_overlay, (0, y), (WINDOW_WIDTH, y), (30, 30, 30), 1)
        
        # Blend pattern dengan opacity rendah
        cv2.addWeighted(pattern_overlay, 0.15, img, 0.85, 0, img)
        
        # Gambar judul utama dengan efek shadow
        title = "DO THE MATH!"
        title_font = cv2.FONT_HERSHEY_TRIPLEX
        title_font_scale = self._get_scaled_font(3.5)
        title_thickness = self._get_scaled_thickness(7)
        
        # Hitung posisi judul agar centered
        title_size = cv2.getTextSize(title, title_font, title_font_scale, title_thickness)[0]
        title_x = (WINDOW_WIDTH - title_size[0]) // 2
        title_y = int(WINDOW_HEIGHT * 0.22)
        
        # Gambar shadow judul (hitam, offset sedikit)
        shadow_offset = int(4 * min_scale)
        cv2.putText(img, title, (title_x + shadow_offset, title_y + shadow_offset), 
                   title_font, title_font_scale, (0, 0, 0), title_thickness + 2)
        
        # Gambar judul asli (cyan/kuning)
        cv2.putText(img, title, (title_x, title_y), 
                   title_font, title_font_scale, (0, 255, 255), title_thickness)
        
        # Font untuk konten lainnya
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Gambar subtitle
        subtitle = "Kuis Matematika - Gambar Jawabanmu di Udara!"
        subtitle_font_scale = self._get_scaled_font(0.9)
        subtitle_thickness = self._get_scaled_thickness(3)
        
        subtitle_size = cv2.getTextSize(subtitle, font, subtitle_font_scale, subtitle_thickness)[0]
        subtitle_x = (WINDOW_WIDTH - subtitle_size[0]) // 2
        subtitle_y = int(WINDOW_HEIGHT * 0.30)
        
        cv2.putText(img, subtitle, (subtitle_x, subtitle_y), 
                   font, subtitle_font_scale, (255, 255, 255), subtitle_thickness)
        
        # Buat kotak instruksi
        box_width = int(WINDOW_WIDTH * 0.6)
        box_height = int(WINDOW_HEIGHT * 0.35)
        box_x = (WINDOW_WIDTH - box_width) // 2
        box_y = int(WINDOW_HEIGHT * 0.38)
        
        # Gambar kotak dengan transparansi
        box_overlay = img.copy()
        cv2.rectangle(box_overlay, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     (50, 50, 50), -1)
        cv2.addWeighted(box_overlay, 0.7, img, 0.3, 0, img)
        
        # Gambar border kotak (cyan)
        cv2.rectangle(img, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     (0, 255, 255), 3)
        
        # Judul instruksi
        inst_font_scale = self._get_scaled_font(0.9)
        inst_thickness = self._get_scaled_thickness(2)
        
        inst_title = "CARA BERMAIN:"
        inst_title_size = cv2.getTextSize(inst_title, font, inst_font_scale + 0.1, 
                                         inst_thickness + 1)[0]
        inst_title_x = box_x + (box_width - inst_title_size[0]) // 2
        inst_title_y = box_y + int(box_height * 0.15)
        
        cv2.putText(img, inst_title, (inst_title_x, inst_title_y),
                   font, inst_font_scale + 0.1, (0, 255, 255), inst_thickness + 1)
        
        # Daftar instruksi dengan icon gambar
        instructions = [
            ("assets/icons/one.png", "1 jari = Menggambar angka"),
            ("assets/icons/four.png", "4 jari = Submit jawaban"),  # Ganti two.png jadi four.png atau sesuaikan
            ("assets/icons/five.png", "5 jari = Hapus gambar"),
        ]

        # Emoji fallback jika icon tidak ditemukan
        emoji_fallback = ["☝", "✋", "🖐"]

        text_x = box_x + int(box_width * 0.35)  # Geser sedikit ke kanan untuk beri ruang icon
        start_y = inst_title_y + int(box_height * 0.2)
        line_spacing = int(box_height * 0.18)
        icon_size = int(40 * min_scale)  # Ukuran icon

        for i, (icon_path, text) in enumerate(instructions):
            y_pos = start_y + (i * line_spacing)
            
            # Posisi icon
            icon_x = text_x - icon_size - int(15 * min_scale)
            icon_y = y_pos - icon_size + int(5 * min_scale)
            
            # Coba load dan tampilkan icon
            try:
                if os.path.exists(icon_path):
                    icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                    
                    if icon is not None:
                        # Resize icon ke ukuran yang diinginkan
                        icon_resized = cv2.resize(icon, (icon_size, icon_size))
                        
                        # Handle icon dengan alpha channel (transparansi)
                        if icon_resized.shape[2] == 4:  # Ada alpha channel
                            # Extract alpha channel
                            alpha = icon_resized[:, :, 3] / 255.0
                            
                            # Blend icon dengan background menggunakan alpha
                            for c in range(3):  # Loop BGR channels
                                img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c] = \
                                    alpha * icon_resized[:, :, c] + \
                                    (1 - alpha) * img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c]
                        else:
                            # Icon tanpa alpha, langsung overlay
                            img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = icon_resized
                    else:
                        # Icon gagal load, pakai emoji fallback
                        cv2.putText(img, emoji_fallback[i], (icon_x, y_pos),
                                font, inst_font_scale * 0.8, (0, 215, 255), inst_thickness)
                else:
                    # File tidak ada, pakai emoji fallback
                    cv2.putText(img, emoji_fallback[i], (icon_x, y_pos),
                            font, inst_font_scale * 0.8, (0, 215, 255), inst_thickness)
            
            except Exception as e:
                # Error saat load icon, pakai emoji fallback
                print(f"⚠ Warning: Gagal load icon {icon_path}: {str(e)}")
                cv2.putText(img, emoji_fallback[i], (icon_x, y_pos),
                        font, inst_font_scale * 0.8, (0, 215, 255), inst_thickness)
            
            # Gambar teks instruksi
            cv2.putText(img, text, (text_x, y_pos),
                    font, inst_font_scale, (255, 255, 255), inst_thickness)
    
    def process_answer(self, canvas: np.ndarray) -> np.ndarray:
        """
        Proses jawaban user: preprocess, predict, check, dan update quiz.
        
        Flow audio:
        - Jawaban BENAR: Play correct sound -> applause (sequential)
        - Jawaban SALAH: Play wrong sound
        
        Args:
            canvas (numpy.ndarray): Canvas berisi gambar jawaban user
            
        Returns:
            numpy.ndarray: Canvas yang di-clear (zeros)
        """
        print("\n" + "="*60)
        print("MEMPROSES JAWABAN...")
        print("="*60)
        
        # Stop audio soal
        self.audio.stop_audio()
        
        if not self.recognizer_loaded:
            print("⚠ Model tidak ter-load!")
            self.ui_manager.show_notification("MODEL NOT LOADED!", color=(0, 0, 255))
            return canvas
        
        # Prediksi digit menggunakan multi-digit recognition
        result, confidence = recognize_multi_digit(self.session, canvas, max_digits=3)
        
        if result is None or confidence < 50:
            print("⚠ Tidak dapat memproses gambar atau confidence terlalu rendah")
            self.ui_manager.show_notification("GAMBAR TIDAK JELAS!", color=(0, 255, 255))
            return canvas
        
        # Convert string result ke integer
        try:
            predicted_answer = int(result)
        except ValueError:
            print(f"⚠ Hasil prediksi tidak valid: {result}")
            self.ui_manager.show_notification("PREDIKSI ERROR!", color=(0, 0, 255))
            return canvas
        
        print(f"Hasil prediksi: {predicted_answer}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Check jawaban
        is_correct = self.quiz.check_answer(predicted_answer)
        
        current_question = self.quiz.get_current_question()
        correct_answer = current_question['correct_answer']
        
        # Set flag bahwa audio jawaban sedang playing
        self.answer_audio_playing = True
        
        if is_correct:
            # BENAR! Play sequential audio: correct -> applause
            print("✓ JAWABAN BENAR!")
            self.ui_manager.show_notification(f"BENAR! Jawaban: {correct_answer}", color=(0, 255, 0))
            self.audio.play_correct_sequence()
            self.confetti.generate_burst(num_particles=150)
        else:
            # SALAH - Play wrong sound only
            print(f"✗ JAWABAN SALAH! Jawaban yang benar: {correct_answer}")
            self.ui_manager.show_notification(f"SALAH! Jawaban: {correct_answer}", color=(0, 0, 255))
            self.audio.play_wrong_sound()
        
        # Pindah ke soal berikutnya
        if not self.quiz.is_finished():
            next_question = self.quiz.next_question()
            if next_question:
                print(f"\n[INFO] Pindah ke {self.quiz.get_progress()}")
                self.pending_audio_path = next_question['audio_path']
            else:
                # Quiz selesai setelah increment
                self.quiz_finished = True
                self.answer_audio_playing = False
                print("\n" + "="*60)
                print("QUIZ SELESAI!")
                print("="*60)
        else:
            # Quiz selesai
            self.quiz_finished = True
            self.answer_audio_playing = False
            print("\n" + "="*60)
            print("QUIZ SELESAI!")
            print("="*60)
        
        # Return cleared canvas
        return np.zeros_like(canvas)
    
    def create_gradient_background(self, img: np.ndarray, x1: int, y1: int, 
                                   x2: int, y2: int, color_top: tuple, color_bottom: tuple):
        """
        Membuat gradient background vertikal.
        
        Args:
            img (numpy.ndarray): Image untuk drawing
            x1, y1: Koordinat top-left corner
            x2, y2: Koordinat bottom-right corner
            color_top (tuple): Warna BGR untuk bagian atas
            color_bottom (tuple): Warna BGR untuk bagian bawah
        """
        height = y2 - y1
        
        for i in range(height):
            ratio = i / height
            
            b = int(color_top[0] * (1 - ratio) + color_bottom[0] * ratio)
            g = int(color_top[1] * (1 - ratio) + color_bottom[1] * ratio)
            r = int(color_top[2] * (1 - ratio) + color_bottom[2] * ratio)
            
            cv2.line(img, (x1, y1 + i), (x2, y1 + i), (b, g, r), 1)
    
    def show_final_score_screen(self, img: np.ndarray):
        """
        Menampilkan final score screen saat quiz selesai.
        
        Args:
            img (numpy.ndarray): Frame untuk display
        """
        # Semi-transparent overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Get final score
        correct, total, percentage = self.quiz.get_score()
        
        # Koordinat box
        box_x1 = 340
        box_y1 = 150
        box_x2 = 940
        box_y2 = 550
        
        # Gradient background
        color_top = (219, 112, 147)
        color_bottom = (130, 0, 139)
        self.create_gradient_background(img, box_x1, box_y1, box_x2, box_y2, 
                                       color_top, color_bottom)
        
        # Border gold tebal
        cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 215, 255), 5)
        
        # Title
        cv2.putText(img, "QUIZ SELESAI!", (430, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 215, 255), 3)
        
        # Score info
        cv2.putText(img, f"Skor Anda: {correct}/{total}", (450, 310), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        cv2.putText(img, f"Persentase: {percentage:.1f}%", (450, 370), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Rating
        if percentage >= 90:
            rating = "EXCELLENT!"
            color = (0, 255, 0)
        elif percentage >= 75:
            rating = "GREAT!"
            color = (0, 255, 255)
        elif percentage >= 60:
            rating = "GOOD!"
            color = (255, 255, 0)
        else:
            rating = "KEEP TRYING!"
            color = (0, 165, 255)
        
        cv2.putText(img, rating, (500, 430), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Instructions
        cv2.putText(img, "Tekan 'R' untuk Restart", (450, 490), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
        cv2.putText(img, "Tekan 'Q' untuk Keluar", (450, 525), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
    
    def run(self):
        """
        Main loop untuk menjalankan aplikasi.
        """
        print("Program dimulai! Tekan 'Q' untuk keluar.\n")
        print("=" * 60)
        print("INTRO SCREEN")
        print("- Tekan SPACE untuk mulai quiz")
        print("=" * 60 + "\n")
        
        while True:
            success, img = self.cap.read()
            
            if not success:
                print("Gagal membaca frame dari webcam")
                break
            
            img = cv2.flip(img, 1)
            
            # Handle intro screen
            if self.show_intro:
                # Play intro audio (sekali saat pertama kali)
                self.play_intro_audio()

                # Tampilkan intro screen
                self.draw_intro_screen(img)
                
                cv2.imshow("DO THE MATH! - Interactive Math Learning", img)
                
                # Check keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # SPACE untuk mulai
                    # Stop intro audio jika masih playing
                    try:
                        import pygame
                        pygame.mixer.music.stop()
                        print("♪ Intro audio stopped")
                    except:
                        pass
                    self.show_intro = False

                    print("\n" + "="*60)
                    print("QUIZ DIMULAI!")
                    print("="*60)
                    print("CARA PENGGUNAAN:")
                    print("- Angkat 1 jari (telunjuk) untuk menggambar")
                    print("- Angkat 4 jari (tanpa jempol) untuk submit jawaban")
                    print("- Angkat 5 jari untuk menghapus canvas")
                    print("="*60 + "\n")
                    
                    # Play audio soal pertama setelah keluar dari intro
                    current_question = self.quiz.get_current_question()
                    if current_question:
                        cv2.waitKey(500)  # Delay sebentar
                        self.audio.play_question_audio(current_question['audio_path'])
                        self.first_audio_played = True
                
                elif key == ord('q'):  # Q untuk quit dari intro
                    # Stop intro audio
                    try:
                        import pygame
                        pygame.mixer.music.stop()
                    except:
                        pass

                    print("\n" + "="*60)
                    print("Program dihentikan dari intro screen.")
                    print("="*60)
                    break
                
                continue  # Skip ke iterasi berikutnya jika masih di intro
            
            # Play audio soal pertama (jika belum dari intro)
            if not self.first_audio_played:
                current_question = self.quiz.get_current_question()
                if current_question:
                    self.audio.play_question_audio(current_question['audio_path'])
                self.first_audio_played = True
            
            # Initialize canvas
            if self.canvas is None:
                self.canvas = np.zeros_like(img)
            
            # Deteksi tangan dan proses gesture
            info = self.gesture_controller.get_hand_info(img)
            fingers = None
            
            # Block gesture jika quiz finished atau audio playing
            is_blocked = self.quiz_finished or self.answer_audio_playing
            
            if info and not is_blocked:
                fingers, _ = info
                current_pos, self.canvas, action = self.gesture_controller.process_gesture(
                    info, self.canvas, img, is_blocked=False
                )
                
                # Handle actions
                if action == "clear":
                    if self.ui_manager.notification_timer == 0:
                        self.ui_manager.show_notification("CANVAS DIHAPUS!", 
                                                        color=(255, 255, 0),
                                                        duration=NOTIFICATION_DURATION // 2)
                        print("\n[INFO] Canvas dihapus!")
                
                elif action == "submit":
                    if self.ui_manager.notification_timer == 0:
                        self.canvas = self.process_answer(self.canvas)
            
            # Gabungkan image dengan canvas
            combined_image = cv2.addWeighted(img, 0.7, self.canvas, 0.3, 0)
            
            # Update audio sequence
            self.audio.update_sequence()
            
            # Update confetti
            if self.confetti.is_active():
                self.confetti.update()
            
            # Check audio jawaban selesai
            if self.answer_audio_playing:
                if not self.audio.is_sequence_playing() and not self.audio.is_playing():
                    self.answer_audio_playing = False
                    
                    if np.any(self.canvas):
                        self.canvas = np.zeros_like(img)
                    
                    if self.pending_audio_path and not self.quiz_finished:
                        self.audio.play_question_audio(self.pending_audio_path)
                        self.pending_audio_path = None
            
            # Render UI
            if not self.quiz_finished:
                # Draw panel kiri (instruksi, score, status)
                self.ui_manager.draw_left_panel(combined_image, self.quiz, self.gesture_controller)
                
                # Draw panel soal di kanan atas
                self.ui_manager.draw_question_panel(combined_image, self.quiz)
                
                # Draw notifikasi
                self.ui_manager.draw_notification(combined_image)
                
                # Draw confetti jika aktif
                if self.confetti.is_active():
                    self.confetti.draw(combined_image)
            else:
                self.show_final_score_screen(combined_image)
            
            # Display
            cv2.imshow("DO THE MATH! - Interactive Math Learning", combined_image)
            
            # Keyboard control
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n" + "="*60)
                print("Program dihentikan.")
                print("="*60)
                break
            elif key == ord('r') and self.quiz_finished:
                self.quiz.reset()
                self.quiz_finished = False
                self.canvas = np.zeros_like(img)
                self.ui_manager.notification_text = ""
                self.ui_manager.notification_timer = 0
                self.confetti.clear()
                self.gesture_controller.reset()
                
                current_question = self.quiz.get_current_question()
                if current_question:
                    self.audio.play_question_audio(current_question['audio_path'])
        
        # Cleanup
        self.audio.cleanup()
        self.cap.release()
        cv2.destroyAllWindows()


# Entry point
if __name__ == "__main__":
    try:
        app = MathQuizApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()