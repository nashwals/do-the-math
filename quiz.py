"""
DO THE MATH! - Interactive Math Quiz dengan Gesture Recognition
Sistem Quiz Management & Audio Player Integration
UPDATED: Support format answers.txt dengan koma (1,8)
"""

import numpy as np
import cv2
import pygame
import os
import json
import threading
import time
from digit_recognition import load_model, recognize_multi_digit

# Import dari gesture_tracking library
from gesture_tracking import (
    DRAW_CHARGE_TIME,
    NOTIFICATION_DURATION,
    BRUSH_SIZE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    DRAWING_AREA_X,
    DRAWING_AREA_WIDTH,
    create_detector,
    create_gesture_state,
    getHandInfo,
    process_gesture,
    draw_status_indicator
)

# ======================= QUIZ MANAGER =======================
class QuizManager:
    def __init__(self, data_folder="data", total_questions=15):
        """
        Mengelola quiz, soal, jawaban, dan audio
        
        Args:
            data_folder: Folder berisi data soal dan audio
            total_questions: Jumlah total soal
        """
        self.data_folder = data_folder
        self.total_questions = total_questions
        self.current_question = 1
        self.score = 0
        self.quiz_completed = False
        
        # Load quiz data
        self.questions = self._load_questions()
        
        # Initialize pygame mixer untuk audio
        pygame.mixer.init()
        self.audio_playing = False
        self.audio_thread = None
        
    def _load_questions(self):
        """
        Load semua soal dan jawaban dari file
        Format: soal_x.png dan audio_soal_x.wav
        """
        questions = {}
        
        for i in range(1, self.total_questions + 1):
            question_img_path = os.path.join(self.data_folder, f"soal_{i}.png")
            audio_path = os.path.join(self.data_folder, f"audio_soal_{i}.wav")
            
            if os.path.exists(question_img_path):
                # Load gambar soal
                question_img = cv2.imread(question_img_path)
                
                # Extract jawaban dari file answer
                answer = self._get_answer_for_question(i)
                
                questions[i] = {
                    'image': question_img,
                    'audio_path': audio_path if os.path.exists(audio_path) else None,
                    'answer': answer
                }
            else:
                print(f"Warning: Soal {i} tidak ditemukan di {question_img_path}")
        
        return questions
    
    def _get_answer_for_question(self, question_num):
        """
        Mendapatkan jawaban untuk soal tertentu
        Mencari di beberapa lokasi: root folder, data folder, current folder
        """
        # Cari di beberapa lokasi
        possible_paths = [
            "answers.txt",                           # Root folder
            os.path.join(self.data_folder, "answers.txt"),  # Data folder
            os.path.join("data", "answers.txt"),     # data folder explicit
            "answers.json",                          # JSON di root
            os.path.join(self.data_folder, "answers.json"), # JSON di data
        ]
        
        for txt_file in possible_paths:
            if os.path.exists(txt_file):
                try:
                    if txt_file.endswith('.txt'):
                        answers_data = self._parse_answers_txt(txt_file)
                    else:
                        with open(txt_file, 'r') as f:
                            answers_data = json.load(f)
                    
                    answer = answers_data.get(str(question_num), None)
                    if answer:
                        return str(answer)
                except Exception as e:
                    print(f"Error reading {txt_file}: {e}")
                    continue
        
        # Jika tidak ada file answers, return None
        print(f"Warning: Answer for question {question_num} not found!")
        return None
    
    def _parse_answers_txt(self, txt_file):
        """
        Parse answers.txt dengan berbagai format
        UPDATED: Support format 1,8 (comma per line)
        
        Returns:
            dict: Dictionary jawaban {question_num: answer}
        """
        import re
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
        
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            print("✗ File answers.txt kosong!")
            return {}
        
        answers = {}
        first_line = lines[0]
        
        print(f"\n📄 Reading answers from: {txt_file}")
        print(f"First line: '{first_line}'")
        
        # Detect format and parse
        # FORMAT 1: "1,8" (comma separated per line) - FORMAT USER!
        if re.match(r'^\d+,\d+', first_line):
            print("✓ Format detected: COMMA PER LINE (1,8)")
            for line in lines:
                parts = line.split(',')
                if len(parts) == 2:
                    num = parts[0].strip()
                    answer = parts[1].strip()
                    if num.isdigit():  # Answer bisa 0, jadi tidak perlu check isdigit untuk answer
                        answers[num] = answer
                        print(f"  Soal {num}: {answer}")
        
        # FORMAT 2: "1. 8" atau "1) 8" atau "1: 8"
        elif re.match(r'^\d+[\.\)\:]\s*\d+', first_line):
            print("✓ Format detected: NUMBERED (1. 8)")
            for line in lines:
                match = re.match(r'^(\d+)[\.\)\:]\s*(\d+)', line)
                if match:
                    answers[match.group(1)] = match.group(2)
                    print(f"  Soal {match.group(1)}: {match.group(2)}")
        
        # FORMAT 3: "1 = 8" atau "1 8"
        elif re.match(r'^\d+\s*[=\s]\s*\d+', first_line):
            print("✓ Format detected: KEY-VALUE (1 = 8)")
            for line in lines:
                parts = line.split('=') if '=' in line else line.split()
                if len(parts) >= 2:
                    num = parts[0].strip()
                    answer = parts[1].strip()
                    if num.isdigit():
                        answers[num] = answer
                        print(f"  Soal {num}: {answer}")
        
        # FORMAT 4: Simple number (urut dari 1)
        elif re.match(r'^\d+$', first_line):
            print("✓ Format detected: SIMPLE (urut dari 1)")
            for i, line in enumerate(lines, start=1):
                if line:  # Tidak harus digit, bisa 0
                    answers[str(i)] = line.strip()
                    print(f"  Soal {i}: {line}")
        
        # FORMAT 5: Comma separated all in one line "8, 12, 5, ..."
        elif ',' in content and len(lines) == 1:
            print("✓ Format detected: COMMA SEPARATED (8, 12, 5)")
            parts = content.split(',')
            for i, part in enumerate(parts, start=1):
                answer = part.strip()
                if answer:
                    answers[str(i)] = answer
                    print(f"  Soal {i}: {answer}")
        
        else:
            print("✗ Format tidak dikenali!")
            print("Supported formats:")
            print("  - 1,8")
            print("  - 1. 8")
            print("  - 1 = 8")
            print("  - 8 (urut)")
        
        print(f"✓ Total parsed: {len(answers)} answers\n")
        return answers
    
    def get_current_question(self):
        """Mendapatkan soal saat ini"""
        if self.current_question in self.questions:
            return self.questions[self.current_question]
        return None
    
    def play_question_audio(self):
        """Memutar audio soal saat ini"""
        question = self.get_current_question()
        
        if question and question['audio_path'] and os.path.exists(question['audio_path']):
            def play_audio():
                try:
                    self.audio_playing = True
                    pygame.mixer.music.load(question['audio_path'])
                    pygame.mixer.music.play()
                    
                    # Tunggu hingga audio selesai
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    
                    self.audio_playing = False
                except Exception as e:
                    print(f"Error playing audio: {e}")
                    self.audio_playing = False
            
            # Play audio di thread terpisah agar tidak blocking
            self.audio_thread = threading.Thread(target=play_audio, daemon=True)
            self.audio_thread.start()
    
    def play_feedback_audio(self, is_correct):
        """
        Memutar audio feedback berdasarkan jawaban
        
        Args:
            is_correct: True jika jawaban benar, False jika salah
        """
        def play_feedback():
            try:
                self.audio_playing = True
                
                if is_correct:
                    # Jawaban benar: correct.wav → applause-cheer.wav
                    correct_path = os.path.join(os.getcwd(), "sound-effects", "correct.wav")
                    applause_path = os.path.join(os.getcwd(), "sound-effects", "applause-cheer.wav")
                    
                    # Fallback paths
                    if not os.path.exists(correct_path):
                        correct_path = os.path.join(self.data_folder, "correct.wav")
                    if not os.path.exists(applause_path):
                        applause_path = os.path.join(self.data_folder, "applause-cheer.wav")
                    
                    if os.path.exists(correct_path):
                        pygame.mixer.music.load(correct_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                    
                    if os.path.exists(applause_path):
                        pygame.mixer.music.load(applause_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                else:
                    # Jawaban salah: wrong.wav
                    wrong_path = os.path.join(os.getcwd(), "sound-effects", "wrong.wav")
                    
                    # Fallback path
                    if not os.path.exists(wrong_path):
                        wrong_path = os.path.join(self.data_folder, "wrong.wav")
                    
                    if os.path.exists(wrong_path):
                        pygame.mixer.music.load(wrong_path)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                
                self.audio_playing = False
            except Exception as e:
                print(f"Error playing feedback audio: {e}")
                self.audio_playing = False
        
        # Play feedback di thread terpisah
        feedback_thread = threading.Thread(target=play_feedback, daemon=True)
        feedback_thread.start()
    
    def check_answer(self, user_answer):
        """
        Mengecek jawaban user
        
        Args:
            user_answer: Jawaban dari user (string)
            
        Returns:
            bool: True jika benar, False jika salah
        """
        question = self.get_current_question()
        
        if question and question['answer']:
            correct_answer = str(question['answer']).strip()
            user_answer = str(user_answer).strip()
            
            print(f"\n🔍 Checking answer:")
            print(f"   User answer: '{user_answer}'")
            print(f"   Correct answer: '{correct_answer}'")
            
            is_correct = (user_answer == correct_answer)
            
            if is_correct:
                self.score += 1
                print("   ✓ CORRECT!")
            else:
                print("   ✗ WRONG!")
            
            return is_correct
        
        print("   ⚠ No answer found for this question!")
        return False
    
    def next_question(self):
        """Pindah ke soal berikutnya"""
        if self.current_question < self.total_questions:
            self.current_question += 1
            
            # Auto-play audio soal berikutnya
            time.sleep(0.5)  # Delay sebentar
            self.play_question_audio()
        else:
            self.quiz_completed = True
    
    def restart_quiz(self):
        """Restart quiz dari awal"""
        self.current_question = 1
        self.score = 0
        self.quiz_completed = False
        self.play_question_audio()


# GestureTracker class removed - using gesture_tracking library functions instead


# ======================= UI MANAGER =======================
class UIManager:
    def __init__(self):
        """Inisialisasi UI Manager"""
        self.notification_text = ""
        self.notification_timer = 0
        self.notification_color = (255, 255, 255)
        
    def draw_left_panel(self, img, quiz_manager, gesture_state):
        """
        Menggambar panel kiri dengan instruksi dan status
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
        progress_text = f"Soal: {quiz_manager.current_question}/{quiz_manager.total_questions}"
        cv2.putText(img, progress_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        score_text = f"Skor: {quiz_manager.score}"
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
            "Tekan 'R' = Restart",
            "Tekan 'N' = Next"
        ]
        
        for instruction in instructions:
            cv2.putText(img, instruction, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 25
        
        # Drawing status indicator
        y_offset += 20
        draw_status_indicator(img, gesture_state, 20, y_offset)
    
    def _draw_status_indicator(self, img, gesture_state, y_pos):
        """Deprecated - using draw_status_indicator from gesture_tracking library"""
        pass
    
    def draw_question_panel(self, img, quiz_manager):
        """
        Menggambar panel soal di bagian atas kanan
        """
        question = quiz_manager.get_current_question()
        
        if question and question['image'] is not None:
            question_img = question['image']
            
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
            
            # Border
            cv2.rectangle(img,
                         (x_pos - padding, y_pos - padding),
                         (x_pos + new_w + padding, y_pos + new_h + padding),
                         (0, 255, 0), 3)
            
            # Paste soal
            img[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = resized_question
    
    def show_notification(self, text, color=(255, 255, 255), duration=NOTIFICATION_DURATION):
        """Menampilkan notifikasi"""
        self.notification_text = text
        self.notification_color = color
        self.notification_timer = duration
    
    def draw_notification(self, img):
        """Menggambar notifikasi di layar"""
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
    
    def draw_quiz_complete(self, img, quiz_manager):
        """Menggambar layar quiz selesai"""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # Title
        cv2.putText(img, "QUIZ SELESAI!", (WINDOW_WIDTH//2 - 200, 200),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 4)
        
        # Score
        score_text = f"Skor Akhir: {quiz_manager.score}/{quiz_manager.total_questions}"
        cv2.putText(img, score_text, (WINDOW_WIDTH//2 - 200, 300),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
        
        # Percentage
        percentage = (quiz_manager.score / quiz_manager.total_questions) * 100
        percentage_text = f"Persentase: {percentage:.1f}%"
        cv2.putText(img, percentage_text, (WINDOW_WIDTH//2 - 180, 370),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Instruction
        cv2.putText(img, "Tekan 'R' untuk restart", (WINDOW_WIDTH//2 - 180, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(img, "Tekan 'Q' untuk keluar", (WINDOW_WIDTH//2 - 180, 500),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


# ======================= MAIN APPLICATION =======================
def main():
    """Main application loop"""
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(3, WINDOW_WIDTH)
    cap.set(4, WINDOW_HEIGHT)
    
    # Initialize components
    try:
        digit_session = load_model()
        print("✓ Digit recognition model loaded")
    except Exception as e:
        print(f"✗ Error loading digit recognition model: {e}")
        return
    
    quiz_manager = QuizManager(data_folder="data", total_questions=15)
    ui_manager = UIManager()
    
    # Initialize gesture tracking from library
    detector = create_detector()
    gesture_state = create_gesture_state()
    
    # Play audio soal pertama
    quiz_manager.play_question_audio()
    
    canvas = None
    last_action = None
    action_cooldown = 0
    
    print("\n" + "="*50)
    print("DO THE MATH! - Quiz Started")
    print("="*50)
    
    while True:
        success, img = cap.read()
        
        if not success:
            break
        
        img = cv2.flip(img, 1)
        
        # Initialize canvas
        if canvas is None:
            canvas = np.zeros_like(img)
        
        # Handle quiz completion
        if quiz_manager.quiz_completed:
            ui_manager.draw_quiz_complete(img, quiz_manager)
            cv2.imshow("DO THE MATH!", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                quiz_manager.restart_quiz()
                canvas = np.zeros_like(img)
            continue
        
        # Get hand info using library function
        hand_info = getHandInfo(detector, img)
        
        if hand_info:
            canvas, action, gesture_state = process_gesture(hand_info, gesture_state, canvas, img)
            
            # Handle actions dengan cooldown
            if action and action != last_action and action_cooldown == 0:
                if action == 'clear':
                    ui_manager.show_notification("Canvas Dihapus!", (0, 255, 255))
                    action_cooldown = 20
                
                elif action == 'submit':
                    # Recognize digit
                    result, confidence = recognize_multi_digit(digit_session, canvas, max_digits=3)
                    
                    if result is not None and confidence > 40:
                        # Check answer
                        is_correct = quiz_manager.check_answer(result)
                        
                        if is_correct:
                            ui_manager.show_notification(
                                f"BENAR! Jawaban: {result}",
                                (0, 255, 0),
                                duration=90
                            )
                            quiz_manager.play_feedback_audio(True)
                            
                            # Auto next question setelah delay
                            threading.Timer(3.0, lambda: quiz_manager.next_question()).start()
                        else:
                            question = quiz_manager.get_current_question()
                            correct_answer = question['answer'] if question else "?"
                            
                            ui_manager.show_notification(
                                f"SALAH! Jawaban: {correct_answer}",
                                (0, 0, 255),
                                duration=90
                            )
                            quiz_manager.play_feedback_audio(False)
                            
                            # Auto next question setelah delay
                            threading.Timer(3.0, lambda: quiz_manager.next_question()).start()
                        
                        # Clear canvas after submit
                        canvas = np.zeros_like(img)
                    else:
                        ui_manager.show_notification("Digit Tidak Jelas!", (0, 165, 255))
                    
                    action_cooldown = 30
                
                last_action = action
        
        # Decrease cooldown
        if action_cooldown > 0:
            action_cooldown -= 1
        
        # Combine images
        combined_img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        
        # Draw UI elements
        ui_manager.draw_left_panel(combined_img, quiz_manager, gesture_state)
        ui_manager.draw_question_panel(combined_img, quiz_manager)
        ui_manager.draw_notification(combined_img)
        
        # Show window
        cv2.imshow("DO THE MATH!", combined_img)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            quiz_manager.restart_quiz()
            canvas = np.zeros_like(img)
            ui_manager.show_notification("Quiz Restart!", (0, 255, 255))
        elif key == ord('n'):
            # Manual next question
            if not quiz_manager.quiz_completed:
                quiz_manager.next_question()
                canvas = np.zeros_like(img)
                ui_manager.show_notification("Soal Berikutnya", (0, 255, 255))
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    
    print("\n" + "="*50)
    print("Quiz Finished!")
    print(f"Final Score: {quiz_manager.score}/{quiz_manager.total_questions}")
    print("="*50)


if __name__ == "__main__":
    main()