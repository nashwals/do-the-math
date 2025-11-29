import cv2
import numpy as np
import time
import pygame
import random
import math

# Import all modules properly
from digit_recognition import load_model, recognize_multi_digit
from game_manager import GameManager
from gesture_tracking import GestureTracking
from ui_overlay import UIOverlay


class DoTheMathGame:
    """Main game class"""
    
    def __init__(self):
        # Setup camera
        self.cap = cv2.VideoCapture(0)
        
        # Request high resolution
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        
        # Get ACTUAL resolution from webcam
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Actual camera resolution: {self.width}x{self.height}")
        
        # Calculate scaling factors
        self.scale_x = self.width / 1280.0
        self.scale_y = self.height / 720.0
        self.min_scale = min(self.scale_x, self.scale_y)
        
        print(f"📏 Scale factors - X: {self.scale_x:.2f}, Y: {self.scale_y:.2f}, Min: {self.min_scale:.2f}")
        
        # Initialize modules
        brush_size = max(15, int(20 * self.min_scale))
        self.gesture_handler = GestureTracking(
            width=self.width,
            height=self.height,
            brush_size=brush_size,
            draw_charge_time=15  # REDUCED from 30 for faster response!
        )
        
        self.ui_overlay = UIOverlay(
            width=self.width,
            height=self.height,
            notification_duration=30
        )
        
        self.game_manager = GameManager(num_questions=5)
        
        # Digit recognizer
        try:
            self.digit_session = load_model()
            print("✅ Digit recognition model loaded")
        except Exception as e:
            print(f"⚠️ Error loading digit model: {e}")
            self.digit_session = None
        
        self.hand_detected_time = None
        self.HAND_DETECTION_THRESHOLD = 1.0
        
        print("✅ Game initialized!")
    
    def get_scaled_font(self, base_size):
        """Get scaled font size"""
        return max(0.4, base_size * self.min_scale)
    
    def get_scaled_thickness(self, base_thickness):
        """Get scaled thickness"""
        return max(1, int(base_thickness * self.min_scale))
    
    def submit_answer(self):
        """Submit jawaban dengan recognition dan visualisasi"""
        if self.digit_session is None:
            self.ui_overlay.show_notification("MODEL NOT LOADED!")
            return
        
        canvas = self.gesture_handler.get_canvas()
        
        # Recognize digit WITH DEBUG VISUALIZATION
        result, confidence = recognize_multi_digit(
            self.digit_session,
            canvas,
            max_digits=2,
            save_debug=True  # ENABLE visualization output!
        )
        
        if result is not None and confidence > 50:
            self.ui_overlay.show_notification(f"ANGKA: {result}")
            
            # Submit ke game manager
            self.game_manager.submit_answer(result)
            print(f"✅ Submitted: {result} ({confidence:.1f}%)")
        else:
            self.ui_overlay.show_notification("TIDAK JELAS!")
            print("⚠️ Digit tidak terdeteksi jelas")
        
        # Reset drawing state
        self.gesture_handler.reset_drawing_state()
    
    def draw_attract_mode(self, img):
        """Draw attract mode screen with enhanced design"""
        # Overlay gelap dengan gradient effect
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # ADD BACKGROUND PATTERN 
        # Draw subtle grid pattern
        pattern_overlay = img.copy()
        grid_spacing = int(80 * self.min_scale)
        
        # Vertical lines
        for x in range(0, self.width, grid_spacing):
            cv2.line(pattern_overlay, (x, 0), (x, self.height), (30, 30, 30), 1)
        
        # Horizontal lines
        for y in range(0, self.height, grid_spacing):
            cv2.line(pattern_overlay, (0, y), (self.width, y), (30, 30, 30), 1)
        
        # Blend pattern (very subtle)
        cv2.addWeighted(pattern_overlay, 0.15, img, 0.85, 0, img)
        
        #  TITLE dengan Shadow Effect (TRIPLEX - special!) 
        title = "DO THE MATH!"
        title_font = cv2.FONT_HERSHEY_TRIPLEX  # TRIPLEX untuk title
        title_font_scale = self.get_scaled_font(3.5)
        title_thickness = self.get_scaled_thickness(7)
        
        # Calculate title position
        title_size = cv2.getTextSize(title, title_font, title_font_scale, title_thickness)[0]
        title_x = (self.width - title_size[0]) // 2
        title_y = int(self.height * 0.22)
        
        # Shadow effect
        shadow_offset = int(4 * self.min_scale)
        cv2.putText(img, title, (title_x + shadow_offset, title_y + shadow_offset), 
                   title_font, title_font_scale, (0, 0, 0), title_thickness + 2)
        
        # Main title
        cv2.putText(img, title, (title_x, title_y), 
                   title_font, title_font_scale, (0, 255, 255), title_thickness)
        
        # FONT PAKAI SIMPLEX
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # SUBTITLE / TAGLINE 
        subtitle = "Kuis Matematika - Gambar Jawabanmu di Udara!"
        subtitle_font_scale = self.get_scaled_font(0.9)
        subtitle_thickness = self.get_scaled_thickness(3)
        
        subtitle_size = cv2.getTextSize(subtitle, font, subtitle_font_scale, subtitle_thickness)[0]
        subtitle_x = (self.width - subtitle_size[0]) // 2
        subtitle_y = int(self.height * 0.30)
        
        cv2.putText(img, subtitle, (subtitle_x, subtitle_y), 
                   font, subtitle_font_scale, (255, 255, 255), subtitle_thickness)
        
        # INSTRUCTION BOX (Transparan 70%) 
        box_width = int(self.width * 0.6)
        box_height = int(self.height * 0.35)
        box_x = (self.width - box_width) // 2
        box_y = int(self.height * 0.38)
        
        # Box dengan transparansi
        box_overlay = img.copy()
        cv2.rectangle(box_overlay, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     (50, 50, 50), -1)
        cv2.addWeighted(box_overlay, 0.7, img, 0.3, 0, img)
        
        # Box border
        cv2.rectangle(img, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height),
                     (0, 255, 255), 3)
        
        # INSTRUCTIONS dengan Icons 
        inst_font_scale = self.get_scaled_font(0.7)
        inst_thickness = self.get_scaled_thickness(3)  # BOLD
        
        # Title instruksi
        inst_title = "CARA BERMAIN:"
        inst_title_size = cv2.getTextSize(inst_title, font, inst_font_scale + 0.1, inst_thickness + 1)[0]
        inst_title_x = box_x + (box_width - inst_title_size[0]) // 2
        inst_title_y = box_y + int(box_height * 0.15)
        
        cv2.putText(img, inst_title, (inst_title_x, inst_title_y),
                   font, inst_font_scale + 0.1, (0, 255, 255), inst_thickness + 1)
        
        # Instructions list dengan icons
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
            
            # Try to load and display icon
            icon_path = f"assets/icons/{icon_name}"
            try:
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon_resized = cv2.resize(icon, (icon_size, icon_size))
                    icon_x = text_x - icon_size - int(10 * self.min_scale)
                    icon_y = y_pos - icon_size + int(5 * self.min_scale)
                    
                    # Handle transparency
                    if icon_resized.shape[2] == 4:
                        alpha = icon_resized[:, :, 3] / 255.0
                        for c in range(3):
                            img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c] = \
                                alpha * icon_resized[:, :, c] + \
                                (1 - alpha) * img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c]
                    else:
                        img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = icon_resized
            except:
                pass  # Icon not found, skip
            
            # Instruction text
            cv2.putText(img, text, (text_x, y_pos),
                       font, inst_font_scale, (255, 255, 255), inst_thickness)
        
        # CALL TO ACTION (Pulsing Effect) 
        cta_text = "[ Tekan SPACE untuk Mulai! ]"
        cta_font_scale = self.get_scaled_font(1.2)
        cta_thickness = self.get_scaled_thickness(4)
        
        # Pulsing effect menggunakan time
        pulse = abs(math.sin(time.time() * 2)) * 0.3 + 0.7 
        cta_color = (int(0 * pulse), int(255 * pulse), int(0 * pulse))
        
        cta_size = cv2.getTextSize(cta_text, font, cta_font_scale, cta_thickness)[0]
        cta_x = (self.width - cta_size[0]) // 2
        cta_y = int(self.height * 0.85)
        
        cv2.putText(img, cta_text, (cta_x, cta_y),
                   font, cta_font_scale, cta_color, cta_thickness)
        pass
    
    def draw_countdown(self, img):
        """Draw countdown screen"""
        # Overlay gelap
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Countdown number
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
            # Start game
            self.game_manager.next_question()
            self.gesture_handler.clear_canvas()
    
    def draw_playing_mode(self, img):
        """Draw playing screen"""
        # Load dan tampilkan PNG soal
        question = self.game_manager.current_question
        if question:
            try:
                question_img = cv2.imread(question['image_path'], cv2.IMREAD_UNCHANGED)
                if question_img is not None:
                    # Ukuran lebih besar (35%)
                    img_size = int(self.width * 0.35)
                    question_resized = cv2.resize(question_img, (img_size, img_size))
                    
                    # Position lebih ke bawah
                    margin = int(self.width * 0.02)
                    x_offset = margin
                    y_offset = int(self.height * 0.12)
                    
                    # Handle transparency
                    if question_resized.shape[2] == 4:
                        alpha = question_resized[:, :, 3] / 255.0
                        for c in range(3):
                            img[y_offset:y_offset+img_size, x_offset:x_offset+img_size, c] = \
                                alpha * question_resized[:, :, c] + \
                                (1 - alpha) * img[y_offset:y_offset+img_size, x_offset:x_offset+img_size, c]
                    else:
                        img[y_offset:y_offset+img_size, x_offset:x_offset+img_size] = question_resized
            except Exception as e:
                print(f"Error loading question image: {e}")
        
        # Draw top bar dengan transparansi
        bar_height = int(self.height * 0.11)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, bar_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
 
        font_scale = self.get_scaled_font(1.0)
        thickness = self.get_scaled_thickness(4) 
        text_y = int(bar_height * 0.65)
        font = cv2.FONT_HERSHEY_SIMPLEX
        bright_white = (255, 255, 255) 
        
        # Score
        score = self.game_manager.score * 20  # Nilai per soal = 20
        score_text = f"Score: {score}"
        score_x = int(self.width * 0.02)  # Pindah ke kiri
        cv2.putText(img, score_text, (score_x, text_y), font, font_scale, bright_white, thickness)
        
        # Question number
        q_num_text = f"Soal {self.game_manager.current_question_index + 1}/{self.game_manager.num_questions}"
        text_size = cv2.getTextSize(q_num_text, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        cv2.putText(img, q_num_text, (text_x, text_y), font, font_scale, bright_white, thickness)
        
        # Timer
        time_remaining = self.game_manager.get_time_remaining()
        timer_color = (0, 255, 0) if time_remaining > 10 else \
                     (0, 255, 255) if time_remaining > 5 else (0, 0, 255)
        
        timer_text = f"Waktu: {time_remaining}s"
        timer_size = cv2.getTextSize(timer_text, font, font_scale, thickness)[0]
        timer_x = self.width - timer_size[0] - int(self.width * 0.02)
        cv2.putText(img, timer_text, (timer_x, text_y), font, font_scale, timer_color, thickness)
        
        # Draw bottom instructions bar 
        bottom_bar_height = int(self.height * 0.083)
        bottom_overlay = img.copy()
        cv2.rectangle(bottom_overlay, (0, self.height - bottom_bar_height), 
                     (self.width, self.height), (30, 30, 30), -1)
        cv2.addWeighted(bottom_overlay, 0.4, img, 0.6, 0, img)  # Same transparency as top bar
        
        # Instructions text
        instructions = "1 jari=Draw | 4 jari=Submit | 5 jari=Clear"
        inst_font_scale = self.get_scaled_font(0.7)
        inst_thickness = self.get_scaled_thickness(4)  
        
        inst_x = int(self.width * 0.02)  # Pojok kiri bawah
        inst_y = self.height - int(bottom_bar_height * 0.35)
        
        cv2.putText(img, instructions, (inst_x, inst_y), font, inst_font_scale, bright_white, inst_thickness)
        
        # Overlay canvas
        canvas = self.gesture_handler.get_canvas()
        combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
        img[:] = combined
        
        # Check time up
        if self.game_manager.is_time_up():
            self.game_manager.submit_answer("0")
    
    def draw_result_mode(self, img):
        """Draw result screen with enhanced visual feedback"""
        overlay = img.copy()
        
        is_correct = self.game_manager.last_answer_correct
        
        if is_correct:
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 255, 0), -1)
            message = "BENAR!"
            color = (0, 255, 0)
            # Draw checkmark shape
            icon_type = "check"
        else:
            cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 255), -1)
            message = "SALAH"
            color = (0, 0, 255)
            # Draw X shape
            icon_type = "cross"
        
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        #  DRAW ICON SHAPE (Checkmark or X) 
        center_x = self.width // 2
        icon_y = int(self.height * 0.28)
        icon_size = int(self.width * 0.12)
        
        if icon_type == "check":
            # Draw checkmark ✓
            # Line 1: Short vertical down-left
            pt1 = (center_x - int(icon_size * 0.3), icon_y)
            pt2 = (center_x - int(icon_size * 0.1), icon_y + int(icon_size * 0.4))
            cv2.line(img, pt1, pt2, color, int(icon_size * 0.15))
            
            # Line 2: Long diagonal up-right
            pt3 = (center_x + int(icon_size * 0.5), icon_y - int(icon_size * 0.3))
            cv2.line(img, pt2, pt3, color, int(icon_size * 0.15))
        else:
            # Draw X
            thickness = int(icon_size * 0.15)
            # Line 1: Top-left to bottom-right
            pt1 = (center_x - int(icon_size * 0.4), icon_y - int(icon_size * 0.4))
            pt2 = (center_x + int(icon_size * 0.4), icon_y + int(icon_size * 0.4))
            cv2.line(img, pt1, pt2, color, thickness)
            
            # Line 2: Top-right to bottom-left
            pt3 = (center_x + int(icon_size * 0.4), icon_y - int(icon_size * 0.4))
            pt4 = (center_x - int(icon_size * 0.4), icon_y + int(icon_size * 0.4))
            cv2.line(img, pt3, pt4, color, thickness)
        
        #  MESSAGE TEXT 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.get_scaled_font(4.5)  # Larger
        thickness = self.get_scaled_thickness(10)  # Extra bold
        
        # Message (centered below icon)
        text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = int(self.height * 0.52)
        cv2.putText(img, message, (text_x, text_y), font, font_scale, color, thickness)
        
        #  ANSWER INFO BOX 
        answer_text = f"Jawaban: {self.game_manager.current_question['answer']}"
        ans_font_scale = self.get_scaled_font(2.0)
        ans_thickness = self.get_scaled_thickness(6)  # Bold
        
        ans_size = cv2.getTextSize(answer_text, font, ans_font_scale, ans_thickness)[0]
        ans_x = (self.width - ans_size[0]) // 2
        ans_y = int(self.height * 0.68)
        cv2.putText(img, answer_text, (ans_x, ans_y), font, ans_font_scale, (255, 255, 255), ans_thickness)
        
        # Auto next
        if self.game_manager.get_state_duration() > 2:
            self.game_manager.current_question_index += 1
            has_next = self.game_manager.next_question()
            
            if has_next:
                self.gesture_handler.clear_canvas()
    
    def draw_score_mode(self, img):
        """Draw final score screen with animations"""
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title - BOLD
        title = "PERMAINAN SELESAI!"
        title_font_scale = self.get_scaled_font(2.2)
        title_thickness = self.get_scaled_thickness(6)
        
        text_size = cv2.getTextSize(title, font, title_font_scale, title_thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = int(self.height * 0.25)
        cv2.putText(img, title, (text_x, text_y), font, title_font_scale, (255, 255, 0), title_thickness)
        
        # Score - BOLD
        score = self.game_manager.score * 20
        total = self.game_manager.num_questions * 20
        score_text = f"SCORE: {score}/{total}"
        
        score_font_scale = self.get_scaled_font(3.5)
        score_thickness = self.get_scaled_thickness(8)
        
        text_size2 = cv2.getTextSize(score_text, font, score_font_scale, score_thickness)[0]
        text_x2 = (self.width - text_size2[0]) // 2
        text_y2 = int(self.height * 0.48)
        cv2.putText(img, score_text, (text_x2, text_y2), font, score_font_scale, (0, 255, 255), score_thickness)
        
        # Grade dengan ICONS
        raw_score = self.game_manager.score
        total_questions = self.game_manager.num_questions
        percentage = (raw_score / total_questions) * 100

        # Determine grade, color, and icons
        if percentage == 100:
            grade = "SEMPURNA!"
            grade_color = (0, 255, 0)
            icon_files = ["stars.png", "trophy.png"]  # Star + Trophy
            animation_type = "confetti"
            audio_file = "victory.mp3"  # UPDATED: Use existing file
        elif percentage >= 70:
            grade = "BAGUS!"
            grade_color = (0, 255, 255)
            icon_files = ["clapping.png"]  # Clapping hands
            animation_type = "clapping"
            audio_file = "clapping.mp3"  # UPDATED: Use existing file
        else:
            grade = "COBA LAGI!"
            grade_color = (0, 165, 255)
            icon_files = ["like.png"]  # Thumbs up
            animation_type = "thumbsup"
            audio_file = "good-job.mp3"  # UPDATED: Use existing file
        
        # RESET animation when first entering score screen
        if not hasattr(self, '_score_screen_initialized') or not self._score_screen_initialized:
            self._animation_particles = []
            self._animation_start_time = time.time()
            self._score_audio_played = False
            self._score_screen_initialized = True
            print("🎬 Animation RESET for new score screen")
        
        # Play audio ONCE when entering score screen
        if not self._score_audio_played:
            try:
                pygame.mixer.Sound(f'assets/audio/{audio_file}').play()
                print(f"🔊 Playing: {audio_file}")
            except:
                print(f"⚠️ Audio not found: {audio_file}")
            self._score_audio_played = True
        
        # Display grade text
        grade_font_scale = self.get_scaled_font(2.2)
        grade_thickness = self.get_scaled_thickness(6)
        
        text_size3 = cv2.getTextSize(grade, font, grade_font_scale, grade_thickness)[0]
        text_x3 = (self.width - text_size3[0]) // 2
        text_y3 = int(self.height * 0.68)
        cv2.putText(img, grade, (text_x3, text_y3), font, grade_font_scale, grade_color, grade_thickness)
        
        # Display icons next to grade
        icon_size = int(self.width * 0.08)
        icon_spacing = int(icon_size * 1.2)
        total_icon_width = len(icon_files) * icon_size + (len(icon_files) - 1) * (icon_spacing - icon_size)
        start_x = text_x3 + text_size3[0] + int(self.width * 0.02)
        icon_y = text_y3 - int(icon_size * 0.7)
        
        for i, icon_file in enumerate(icon_files):
            icon_x = start_x + (i * icon_spacing)
            icon_path = f"assets/icons/{icon_file}"
            
            try:
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon_resized = cv2.resize(icon, (icon_size, icon_size))
                    
                    # Handle transparency
                    if icon_resized.shape[2] == 4:
                        alpha = icon_resized[:, :, 3] / 255.0
                        for c in range(3):
                            img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c] = \
                                alpha * icon_resized[:, :, c] + \
                                (1 - alpha) * img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size, c]
                    else:
                        img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = icon_resized
            except Exception as e:
                print(f"⚠️ Icon not found: {icon_file}")
        
        #  ANIMATION OVERLAY 
        self._draw_score_animation(img, animation_type)
        
        # Thank you
        thanks = "TERIMA KASIH!"
        thanks_font_scale = self.get_scaled_font(1.6)
        thanks_thickness = self.get_scaled_thickness(5)
        
        text_size4 = cv2.getTextSize(thanks, font, thanks_font_scale, thanks_thickness)[0]
        text_x4 = (self.width - text_size4[0]) // 2
        text_y4 = int(self.height * 0.82)
        cv2.putText(img, thanks, (text_x4, text_y4), font, thanks_font_scale, (255, 255, 255), thanks_thickness)
        
        # Auto return
        if self.game_manager.get_state_duration() > 5:
            # Stop any playing audio before returning
            pygame.mixer.music.stop()
            pygame.mixer.stop()  # Stop all sound effects too
            
            self._score_screen_initialized = False  # Reset flag for next time
            self.game_manager.return_to_attract()
    
    def _draw_score_animation(self, img, animation_type):
        """Draw animated particles based on score with improved physics"""
        # Initialize animation particles if not exists
        if not hasattr(self, '_animation_particles'):
            self._animation_particles = []
            self._animation_start_time = time.time()
        
        duration = time.time() - self._animation_start_time
        
        # Generate new particles
        if duration < 3.0:  # Animate for 3 seconds
            if animation_type == "confetti":
                # Reduced confetti count for better performance
                for _ in range(3):  # REDUCED from 8 to 3
                    # Spawn from top, spread across width
                    x = random.randint(0, self.width)
                    y = -20  # Start above screen
                    vx = random.uniform(-3, 3)
                    vy = random.uniform(2, 6)
                    rotation = random.uniform(0, 360)
                    rotation_speed = random.uniform(-15, 15)
                    
                    color = random.choice([
                        (255, 69, 0),    # Red-Orange
                        (255, 215, 0),   # Gold
                        (50, 205, 50),   # Lime Green
                        (30, 144, 255),  # Dodger Blue
                        (218, 112, 214), # Orchid
                        (255, 20, 147),  # Deep Pink
                        (0, 255, 255),   # Cyan
                    ])
                    
                    self._animation_particles.append({
                        'x': x, 'y': y, 'vx': vx, 'vy': vy,
                        'color': color,
                        'size': random.randint(8, 14),  # Slightly larger
                        'rotation': rotation,
                        'rotation_speed': rotation_speed,
                        'type': 'confetti'
                    })
            
            elif animation_type == "clapping":
                # Reduced clapping hands - cleaner look
                for _ in range(1):  # REDUCED from 3 to 1
                    side = random.choice(['left', 'right'])
                    if side == 'left':
                        x = -60
                        y = random.randint(int(self.height * 0.2), int(self.height * 0.8))
                        vx = random.uniform(5, 9)
                        vy = random.uniform(-1, 1)
                        rotation = random.uniform(-30, 30)
                    else:  # right
                        x = self.width + 60
                        y = random.randint(int(self.height * 0.2), int(self.height * 0.8))
                        vx = random.uniform(-9, -5)
                        vy = random.uniform(-1, 1)
                        rotation = random.uniform(150, 210)
                    
                    self._animation_particles.append({
                        'x': x, 'y': y, 'vx': vx, 'vy': vy,
                        'icon': 'clapping.png',
                        'size': random.randint(55, 75),  # Slightly larger
                        'rotation': rotation,
                        'type': 'icon'
                    })
            
            elif animation_type == "thumbsup":
                # Reduced thumbs up - cleaner look
                for _ in range(1):  # REDUCED from 3 to 1
                    side = random.choice(['left', 'right', 'bottom'])
                    if side == 'left':
                        x = -50
                        y = random.randint(int(self.height * 0.3), int(self.height * 0.7))
                        vx = random.uniform(4, 7)
                        vy = random.uniform(-2, 2)
                        rotation = random.uniform(-30, 30)
                    elif side == 'right':
                        x = self.width + 50
                        y = random.randint(int(self.height * 0.3), int(self.height * 0.7))
                        vx = random.uniform(-7, -4)
                        vy = random.uniform(-2, 2)
                        rotation = random.uniform(150, 210)
                    else:  # bottom
                        x = random.randint(int(self.width * 0.2), int(self.width * 0.8))
                        y = self.height + 50
                        vx = random.uniform(-2, 2)
                        vy = random.uniform(-8, -5)
                        rotation = random.uniform(0, 360)
                    
                    self._animation_particles.append({
                        'x': x, 'y': y, 'vx': vx, 'vy': vy,
                        'icon': 'like.png',
                        'size': random.randint(50, 70),  # Slightly larger
                        'rotation': rotation,
                        'type': 'icon'
                    })
        
        # Update and draw particles
        new_particles = []
        for particle in self._animation_particles:
            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Physics
            if particle['type'] == 'confetti':
                # Gravity and wind
                particle['vy'] += 0.4  # Stronger gravity
                particle['vx'] += random.uniform(-0.2, 0.2)  # Wind effect
                particle['rotation'] += particle['rotation_speed']
            
            # Check if still on screen (with margin)
            margin = 100
            if (-margin <= particle['x'] <= self.width + margin and 
                -margin <= particle['y'] <= self.height + margin):
                
                x, y = int(particle['x']), int(particle['y'])
                
                if particle['type'] == 'confetti':
                    # Draw rotated confetti (rectangle)
                    size = particle['size']
                    angle = particle['rotation']
                    
                    # Create rotated rectangle points
                    rad = math.radians(angle)
                    cos_a, sin_a = math.cos(rad), math.sin(rad)
                    
                    # Rectangle corners
                    w, h = size, size * 2
                    corners = [
                        (-w/2, -h/2), (w/2, -h/2),
                        (w/2, h/2), (-w/2, h/2)
                    ]
                    
                    # Rotate and translate
                    rotated = []
                    for cx, cy in corners:
                        rx = cx * cos_a - cy * sin_a + x
                        ry = cx * sin_a + cy * cos_a + y
                        rotated.append((int(rx), int(ry)))
                    
                    # Draw filled polygon
                    pts = np.array(rotated, np.int32)
                    cv2.fillPoly(img, [pts], particle['color'])
                
                elif particle['type'] == 'icon':
                    # Draw icon with rotation
                    try:
                        icon = cv2.imread(f"assets/icons/{particle['icon']}", cv2.IMREAD_UNCHANGED)
                        if icon is not None:
                            size = particle['size']
                            icon_resized = cv2.resize(icon, (size, size))
                            
                            # Rotate icon
                            angle = particle.get('rotation', 0)
                            center = (size // 2, size // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            icon_rotated = cv2.warpAffine(icon_resized, M, (size, size))
                            
                            # Place on screen with bounds check
                            y1, y2 = max(0, y), min(self.height, y + size)
                            x1, x2 = max(0, x), min(self.width, x + size)
                            
                            iy1, iy2 = max(0, -y), size - max(0, (y + size) - self.height)
                            ix1, ix2 = max(0, -x), size - max(0, (x + size) - self.width)
                            
                            if y2 > y1 and x2 > x1 and iy2 > iy1 and ix2 > ix1:
                                icon_crop = icon_rotated[iy1:iy2, ix1:ix2]
                                
                                if icon_crop.shape[2] == 4:
                                    alpha = icon_crop[:, :, 3] / 255.0
                                    for c in range(3):
                                        img[y1:y2, x1:x2, c] = \
                                            alpha * icon_crop[:, :, c] + \
                                            (1 - alpha) * img[y1:y2, x1:x2, c]
                    except Exception as e:
                        pass  # Icon error, skip this particle
                
                new_particles.append(particle)
        
        self._animation_particles = new_particles
    
    def run(self):
        """Main game loop"""
        print("🎮 Starting DO THE MATH...")
        print(f"📺 Resolution: {self.width}x{self.height}")
        print(f"📏 Scale factor: {self.min_scale:.2f}x")
        
        # Play intro audio
        self.game_manager.play_intro_audio()
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("⚠️ Failed to read from camera")
                break
            
            img = cv2.flip(img, 1)
            
            # Get hand info
            info = self.gesture_handler.get_hand_info(img)
            
            # State machine
            state = self.game_manager.state
            
            if state == "ATTRACT":
                self.draw_attract_mode(img)

                if info is not None:
                    pass
            
            elif state == "COUNTDOWN":
                self.draw_countdown(img)
            
            elif state == "PLAYING":
                self.draw_playing_mode(img)
                
                # Handle gesture
                if info:
                    gesture_status = self.gesture_handler.handle_gesture(img, info)
                    
                    # Handle submit action
                    if gesture_status['action'] == 'submit':
                        self.submit_answer()
                    elif gesture_status['action'] == 'clear':
                        self.ui_overlay.show_notification("Hapus Canvas")
                    self.ui_overlay.display_ready_button(img, gesture_status)
                
                # Display notification
                self.ui_overlay.display_notification(img)
            
            elif state == "RESULT":
                self.draw_result_mode(img)
            
            elif state == "SCORE":
                self.draw_score_mode(img)
            
            # Display
            cv2.imshow("DO THE MATH", img)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # SPACEBAR untuk start game
                if state == "ATTRACT":
                    # Stop opening audio to prevent conflict
                    pygame.mixer.music.stop()
                    
                    self.game_manager.start_game()
                    print("🎮 Game started by SPACEBAR!")
            elif key == ord('d'):  # Debug key
                print(f"State: {state}, Resolution: {self.width}x{self.height}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("👋 Game closed!")


if __name__ == "__main__":
    try:
        game = DoTheMathGame()
        game.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()