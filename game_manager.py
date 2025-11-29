"""
Game Manager untuk DO THE MATH
Mengelola state, soal, scoring, dan audio
"""

import pygame
import random
import json
import time
from pathlib import Path


class GameManager:
    """Mengelola seluruh game flow dan state"""
    
    def __init__(self, num_questions=3):
        # Game state
        self.state = "ATTRACT"  # ATTRACT, COUNTDOWN, PLAYING, RESULT, SCORE
        self.num_questions = num_questions
        self.current_question_index = 0
        self.score = 0
        
        # Questions
        self.questions = []
        self.current_question = None
        
        # Timing
        self.state_start_time = time.time()
        self.drawing_start_time = None
        self.drawing_time_limit = 30  # detik
        
        # Audio
        pygame.mixer.init()
        self.current_audio = None
        
        # Load questions
        self._load_questions()
        
    def _load_questions(self):
        """Load semua soal dari data folder"""
        data_path = Path("data")
        
        # Load answers dari file txt
        answers = {}
        try:
            with open(data_path / "answers.txt", 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines, start=1):
                    answers[i] = int(line.strip())
        except Exception as e:
            print(f"Error loading answers.txt: {e}")
            return
        
        all_questions = []
        for i in range(1, 16):
            question_data = {
                'id': i,
                'image_path': str(data_path / f"soal_{i}.png"),
                'audio_path': str(data_path / f"audio_soal_{i}.wav"),
                'answer': answers.get(i, 0)
            }
            all_questions.append(question_data)
        
        # Random pilih soal
        self.questions = random.sample(all_questions, self.num_questions)
        
    def start_game(self):
        """Mulai game dari awal"""
        self.state = "COUNTDOWN"
        self.current_question_index = 0
        self.score = 0
        self.state_start_time = time.time()
        
    def next_question(self):
        """Load soal berikutnya"""
        if self.current_question_index < len(self.questions):
            self.current_question = self.questions[self.current_question_index]
            self.state = "PLAYING"
            self.drawing_start_time = time.time()
            self.state_start_time = time.time()
            
            # Play audio soal
            self.play_question_audio()
            return True
        else:
            # Game selesai
            self.state = "SCORE"
            self.state_start_time = time.time()
            return False
    
    def play_question_audio(self):
        """Play audio narasi soal"""
        if self.current_question:
            audio_path = self.current_question['audio_path']
            try:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"Error playing audio: {e}")
    
    def play_intro_audio(self):
        """Play audio intro"""
        try:
            pygame.mixer.music.load('assets/audio/opening.wav')
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing intro audio: {e}")
    
    def play_feedback_audio(self, is_correct):
        """Play audio feedback"""
        try:
            if is_correct:
                pygame.mixer.music.load('assets/audio/correct.wav')
            else:
                pygame.mixer.music.load('assets/wrong.wav')
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing feedback audio: {e}")
    
    def submit_answer(self, user_answer):
        """Submit jawaban user dan cek benar/salah"""
        if self.current_question is None:
            return False
        
        correct_answer = self.current_question['answer']
        
        try:
            user_answer_int = int(user_answer)
            is_correct = (user_answer_int == correct_answer)
            
            if is_correct:
                self.score += 1
            
            # Transition ke result
            self.state = "RESULT"
            self.state_start_time = time.time()
            self.last_answer_correct = is_correct
            self.user_answer = user_answer_int
            
            # Play feedback
            self.play_feedback_audio(is_correct)
            
            return is_correct
            
        except ValueError:
            return False
    
    def get_time_remaining(self):
        """Get sisa waktu drawing"""
        if self.drawing_start_time is None:
            return self.drawing_time_limit
        
        elapsed = time.time() - self.drawing_start_time
        remaining = max(0, self.drawing_time_limit - elapsed)
        return int(remaining)
    
    def is_time_up(self):
        """Check apakah waktu habis"""
        return self.get_time_remaining() == 0
    
    def get_state_duration(self):
        """Get berapa lama di state saat ini"""
        return time.time() - self.state_start_time
    
    def return_to_attract(self):
        """Kembali ke attract mode"""
        self.state = "ATTRACT"
        self.state_start_time = time.time()
        self._load_questions()  # Shuffle questions baru
        
        # Play intro audio
        self.play_intro_audio()