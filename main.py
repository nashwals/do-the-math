import cv2
import numpy as np
import time
import pygame
import math

# Import modules
from modules.digit_recognition import load_model, recognize_multi_digit
from modules.game_manager import GameManager
from modules.gesture_tracking import GestureHandler
from modules.ui_overlay import UIOverlay


class DoTheMathGame:
    """Class utama game - FOKUS INTRO SCREEN"""
    
    def __init__(self):
        # Setup kamera
        self.cap = cv2.VideoCapture(0)
        
        # Minta resolusi tinggi ke kamera
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        
        # Ambil resolusi asli dari webcam
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Resolusi kamera: {self.width}x{self.height}")
        
        # Hitung faktor scaling buat responsive design
        self.scale_x = self.width / 1280.0
        self.scale_y = self.height / 720.0
        self.min_scale = min(self.scale_x, self.scale_y)
        
        print(f"Faktor skala - X: {self.scale_x:.2f}, Y: {self.scale_y:.2f}")
        
        # Inisialisasi modules
        brush_size = max(15, int(20 * self.min_scale))
        self.gesture_handler = GestureHandler(
            width=self.width,
            height=self.height,
            brush_size=brush_size,
            draw_charge_time=15
        )
        
        self.ui_overlay = UIOverlay(
            width=self.width,
            height=self.height,
            notification_duration=30
        )
        
        self.game_manager = GameManager(num_questions=5)
        
        # Load model digit recognition
        try:
            self.digit_session = load_model()
            print("Model digit recognition berhasil dimuat")
        except Exception as e:
            print(f"Error loading digit model: {e}")
            self.digit_session = None
        
        print("Game berhasil diinisialisasi!")
    
    def get_scaled_font(self, base_size):
        """Mengambil ukuran font yang sudah discale"""
        return max(0.4, base_size * self.min_scale)
    
    def get_scaled_thickness(self, base_thickness):
        """Mengambil ketebalan garis yang sudah discale"""
        return max(1, int(base_thickness * self.min_scale))
    
    # INTRO GAME SCREEN - KONTRIBUSI UTAMA LU
    def draw_attract_mode(self, img):
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
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Bikin background pattern grid yang halus
        pattern_overlay = img.copy()
        grid_spacing = int(80 * self.min_scale)
        
        # Gambar garis vertikal
        for x in range(0, self.width, grid_spacing):
            cv2.line(pattern_overlay, (x, 0), (x, self.height), (30, 30, 30), 1)
        
        # Gambar garis horizontal
        for y in range(0, self.height, grid_spacing):
            cv2.line(pattern_overlay, (0, y), (self.width, y), (30, 30, 30), 1)
        
        # Blend pattern dengan opacity rendah biar ga terlalu kelihatan
        cv2.addWeighted(pattern_overlay, 0.15, img, 0.85, 0, img)
        
        # Gambar judul utama dengan efek shadow
        title = "DO THE MATH!"
        title_font = cv2.FONT_HERSHEY_TRIPLEX  # Font khusus buat judul
        title_font_scale = self.get_scaled_font(3.5)
        title_thickness = self.get_scaled_thickness(7)
        
        # Hitung posisi judul biar di tengah atas
        title_size = cv2.getTextSize(title, title_font, title_font_scale, title_thickness)[0]
        title_x = (self.width - title_size[0]) // 2
        title_y = int(self.height * 0.22)
        
        # Gambar bayangan judul dulu (warna hitam, offset dikit)
        shadow_offset = int(4 * self.min_scale)
        cv2.putText(img, title, (title_x + shadow_offset, title_y + shadow_offset), 
                   title_font, title_font_scale, (0, 0, 0), title_thickness + 2)
        
        # Gambar judul asli di atas bayangan (warna cyan)
        cv2.putText(img, title, (title_x, title_y), 
                   title_font, title_font_scale, (0, 255, 255), title_thickness)
        
        # Sisanya pakai font SIMPLEX yang lebih jelas
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Gambar subtitle atau tagline
        subtitle = "Kuis Matematika - Gambar Jawabanmu di Udara!"
        subtitle_font_scale = self.get_scaled_font(0.9)
        subtitle_thickness = self.get_scaled_thickness(3)
        
        subtitle_size = cv2.getTextSize(subtitle, font, subtitle_font_scale, subtitle_thickness)[0]
        subtitle_x = (self.width - subtitle_size[0]) // 2
        subtitle_y = int(self.height * 0.30)
        
        cv2.putText(img, subtitle, (subtitle_x, subtitle_y), 
                   font, subtitle_font_scale, (255, 255, 255), subtitle_thickness)
        
        # Bikin kotak instruksi dengan transparansi tujuh puluh persen
        box_width = int(self.width * 0.6)
        box_height = int(self.height * 0.35)
        box_x = (self.width - box_width) // 2
        box_y = int(self.height * 0.38)
        
        # Gambar kotak dengan transparansi
        box_overlay = img.copy()
        cv2.rectangle(box_overlay, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     (50, 50, 50), -1)
        cv2.addWeighted(box_overlay, 0.7, img, 0.3, 0, img)
        
        # Gambar border kotak (warna cyan)
        cv2.rectangle(img, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     (0, 255, 255), 3)
        
        # Gambar judul instruksi
        inst_font_scale = self.get_scaled_font(0.7)
        inst_thickness = self.get_scaled_thickness(3)
        
        inst_title = "CARA BERMAIN:"
        inst_title_size = cv2.getTextSize(inst_title, font, inst_font_scale + 0.1, inst_thickness + 1)[0]
        inst_title_x = box_x + (box_width - inst_title_size[0]) // 2
        inst_title_y = box_y + int(box_height * 0.15)
        
        cv2.putText(img, inst_title, (inst_title_x, inst_title_y),
                   font, inst_font_scale + 0.1, (0, 255, 255), inst_thickness + 1)
        
        # Daftar instruksi dengan icon
        instructions = [
            ("one.png", "1 jari = Menggambar angka"),
            ("two.png", "4 jari = Submit jawaban"),
            ("five.png", "5 jari = Hapus gambar"),
        ]
        
        icon_size = int(40 * self.min_scale)
        text_x = box_x + int(box_width * 0.35)
        start_y = inst_title_y + int(box_height * 0.2)
        line_spacing = int(box_height * 0.18)
        
        for i, (icon_name, text) in enumerate(instructions):
            y_pos = start_y + (i * line_spacing)
            
            # Coba load dan tampilkan icon
            icon_path = f"assets/icons/{icon_name}"
            try:
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon_resized = cv2.resize(icon, (icon_size, icon_size))
                    icon_x = text_x - icon_size - int(10 * self.min_scale)
                    icon_y = y_pos - icon_size + int(5 * self.min_scale)
                    
                    # Handle transparansi icon (alpha channel)
                    if icon_resized.shape[2] == 4:
                        alpha = icon_resized[:, :, 3] / 255.0
                        for c in range(3):
                            img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c] = \
                                alpha * icon_resized[:, :, c] + \
                                (1 - alpha) * img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c]
                    else:
                        img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = icon_resized
            except:
                pass  # Kalo icon ga ketemu, skip aja
            
            # Gambar teks instruksi
            cv2.putText(img, text, (text_x, y_pos),
                       font, inst_font_scale, (255, 255, 255), inst_thickness)
        
        # Gambar call to action yang berkedip
        cta_text = "[ Tekan SPACE untuk Mulai! ]"
        cta_font_scale = self.get_scaled_font(1.2)
        cta_thickness = self.get_scaled_thickness(4)
        
        # Efek berkedip pakai sine wave
        pulse = abs(math.sin(time.time() * 2)) * 0.3 + 0.7  # Range: 0.7 sampe 1.0
        cta_color = (int(0 * pulse), int(255 * pulse), int(0 * pulse))
        
        cta_size = cv2.getTextSize(cta_text, font, cta_font_scale, cta_thickness)[0]
        cta_x = (self.width - cta_size[0]) // 2
        cta_y = int(self.height * 0.85)
        
        cv2.putText(img, cta_text, (cta_x, cta_y),
                   font, cta_font_scale, cta_color, cta_thickness)
    
    # STATE IN GAME
    def draw_countdown(self, img):
        """Gambar layar countdown (3-2-1)"""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        duration = self.game_manager.get_state_duration()
        countdown = max(0, 3 - int(duration))
        
        if countdown > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.get_scaled_font(10.0)
            thickness = self.get_scaled_thickness(20)
            
            text = str(countdown)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (self.width - text_size[0]) // 2
            text_y = (self.height + text_size[1]) // 2
            
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        else:
            # Mulai game
            self.game_manager.next_question()
            self.gesture_handler.clear_canvas()
    
    def draw_playing_mode(self, img):
        """Gambar layar bermain (simplified)"""
        # Load gambar soal
        question = self.game_manager.current_question
        if question:
            try:
                question_img = cv2.imread(question['image_path'], cv2.IMREAD_UNCHANGED)
                if question_img is not None:
                    img_size = int(self.width * 0.35)
                    question_resized = cv2.resize(question_img, (img_size, img_size))
                    
                    margin = int(self.width * 0.02)
                    x_offset = margin
                    y_offset = int(self.height * 0.12)
                    
                    # Handle transparansi gambar
                    if question_resized.shape[2] == 4:
                        alpha = question_resized[:, :, 3] / 255.0
                        for c in range(3):
                            img[y_offset:y_offset+img_size, x_offset:x_offset+img_size, c] = \
                                alpha * question_resized[:, :, c] + \
                                (1 - alpha) * img[y_offset:y_offset+img_size, x_offset:x_offset+img_size, c]
            except Exception as e:
                print(f"Error loading question: {e}")
        
        # Gambar bar atas
        bar_height = int(self.height * 0.11)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, bar_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.get_scaled_font(1.0)
        thickness = self.get_scaled_thickness(4)
        text_y = int(bar_height * 0.65)
        
        # Tampilkan score
        score_text = f"Score: {self.game_manager.score * 20}"
        cv2.putText(img, score_text, (int(self.width * 0.02), text_y), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Tampilkan nomor soal
        q_text = f"Soal {self.game_manager.current_question_index + 1}/{self.game_manager.num_questions}"
        text_size = cv2.getTextSize(q_text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        cv2.putText(img, q_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Tampilkan timer
        time_remaining = self.game_manager.get_time_remaining()
        timer_color = (0, 255, 0) if time_remaining > 10 else \
                     (0, 255, 255) if time_remaining > 5 else (0, 0, 255)
        timer_text = f"Waktu: {time_remaining}s"
        timer_size = cv2.getTextSize(timer_text, font, font_scale, thickness)[0]
        timer_x = self.width - timer_size[0] - int(self.width * 0.02)
        cv2.putText(img, timer_text, (timer_x, text_y), font, font_scale, timer_color, thickness)
        
        # Overlay canvas gambar
        canvas = self.gesture_handler.get_canvas()
        combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        img[:] = combined
        
        # Cek apakah waktu habis
        if self.game_manager.is_time_up():
            self.game_manager.submit_answer("0")
    
    def draw_result_mode(self, img):
        """Gambar layar hasil (simple)"""
        overlay = img.copy()
        
        is_correct = self.game_manager.last_answer_correct
        
        if is_correct:
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 255, 0), -1)
            message = "BENAR!"
            color = (0, 255, 0)
        else:
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 255), -1)
            message = "SALAH"
            color = (0, 0, 255)
        
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.get_scaled_font(4.5)
        thickness = self.get_scaled_thickness(10)
        
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = int(self.height * 0.52)
        cv2.putText(img, message, (text_x, text_y), font, font_scale, color, thickness)
        
        # Auto next setelah dua detik
        if self.game_manager.get_state_duration() > 2:
            self.game_manager.current_question_index += 1
            has_next = self.game_manager.next_question()
            
            if has_next:
                self.gesture_handler.clear_canvas()
            else:
                # Game selesai, balik ke attract
                print("Game selesai, balik ke attract mode")
                self.game_manager.return_to_attract()
    
    def submit_answer(self):
        """Submit jawaban pakai digit recognition"""
        if self.digit_session is None:
            self.ui_overlay.show_notification("MODEL NOT LOADED!")
            return
        
        canvas = self.gesture_handler.get_canvas()
        
        # Recognize digit
        result, confidence = recognize_multi_digit(
            self.digit_session,
            canvas,
            max_digits=2
        )
        
        if result is not None and confidence > 50:
            self.ui_overlay.show_notification(f"ANGKA: {result}")
            self.game_manager.submit_answer(result)
            print(f"Jawaban tersubmit: {result} ({confidence:.1f}%)")
        else:
            self.ui_overlay.show_notification("TIDAK JELAS!")
            print("Digit ga terdeteksi jelas")
        
        self.gesture_handler.reset_drawing_state()
    
    # MAIN GAME LOOP
    def run(self):
        """Main game loop"""
        print("Starting DO THE MATH (INTRO FOCUS VERSION)...")
        print(f"Resolusi: {self.width}x{self.height}")
        
        # Play audio intro
        self.game_manager.play_intro_audio()
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("Gagal baca dari kamera")
                break
            
            img = cv2.flip(img, 1)
            
            # Ambil info tangan
            info = self.gesture_handler.get_hand_info(img)
            
            # State machine
            state = self.game_manager.state
            
            if state == "ATTRACT":
                # KONTRIBUSI UTAMA LU
                self.draw_attract_mode(img)
            
            elif state == "COUNTDOWN":
                self.draw_countdown(img)
            
            elif state == "PLAYING":
                self.draw_playing_mode(img)
                
                if info:
                    gesture_status = self.gesture_handler.handle_gesture(img, info)
                    
                    if gesture_status['action'] == 'submit':
                        self.submit_answer()
                    elif gesture_status['action'] == 'clear':
                        self.ui_overlay.show_notification("Hapus Canvas")
                    
                    self.ui_overlay.display_ready_button(img, gesture_status)
                
                self.ui_overlay.display_notification(img)
            
            elif state == "RESULT":
                self.draw_result_mode(img)
            
            # Tampilkan ke layar
            cv2.imshow("DO THE MATH", img)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if state == "ATTRACT":
                    pygame.mixer.music.stop()
                    self.game_manager.start_game()
                    print("Game dimulai!")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("Game ditutup!")


if __name__ == "__main__":
    try:
        game = DoTheMathGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()