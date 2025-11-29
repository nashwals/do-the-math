"""
Quiz Manager Module untuk DO THE MATH!
======================================

Modul ini mengelola:
1. Loading soal dari folder data/
2. Loading jawaban dari answers.txt
3. Randomisasi urutan soal
4. Tracking progress dan scoring
5. Validasi jawaban user

Author: [Nama Anda]
Date: November 2025
"""

import os
import random
from typing import List, Dict, Tuple, Optional


class QuizManager:
    """
    Class untuk mengelola quiz matematika.
    
    Attributes:
        questions (List[Dict]): List soal yang sudah di-load
        current_index (int): Index soal yang sedang aktif
        correct_count (int): Jumlah jawaban benar
        total_answered (int): Jumlah soal yang sudah dijawab
    """
    
    def __init__(self, data_folder: str = "data", answers_file: str = "answers.txt"):
        """
        Inisialisasi Quiz Manager.
        
        Args:
            data_folder (str): Path ke folder yang berisi soal dan audio
            answers_file (str): Path ke file yang berisi jawaban
        """
        self.data_folder = data_folder
        self.answers_file = answers_file
        self.questions: List[Dict] = []
        self.current_index: int = 0
        self.correct_count: int = 0
        self.total_answered: int = 0
        
        # Load dan setup quiz
        self._load_questions()
        self._load_answers()
        self._validate_questions()
        self._randomize_questions()
        
        print(f"\n✓ Quiz Manager initialized: {len(self.questions)} soal siap!")
    
    def _load_questions(self):
        """
        Load semua soal dari folder data/.
        
        Scan folder untuk file soal_*.png dan buat mapping ke audio_soal_*.wav
        """
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Folder '{self.data_folder}' tidak ditemukan!")
        
        # Scan untuk file soal_*.png
        image_files = [f for f in os.listdir(self.data_folder) 
                      if f.startswith('soal_') and f.endswith('.png')]
        
        if len(image_files) == 0:
            raise FileNotFoundError(f"Tidak ada file soal ditemukan di '{self.data_folder}'!")
        
        # Sort berdasarkan nomor soal
        image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        for img_file in image_files:
            # Ekstrak nomor soal dari filename
            # Format: soal_1.png -> 1
            question_num = int(img_file.split('_')[1].split('.')[0])
            
            # Mapping ke audio file
            audio_file = f"audio_soal_{question_num}.wav"
            
            image_path = os.path.join(self.data_folder, img_file)
            audio_path = os.path.join(self.data_folder, audio_file)
            
            # Buat question object
            question = {
                'id': question_num,
                'image_path': image_path,
                'audio_path': audio_path,
                'correct_answer': None  # Akan di-load dari answers.txt
            }
            
            self.questions.append(question)
        
        print(f"✓ Loaded {len(self.questions)} soal dari folder '{self.data_folder}'")
    
    def _load_answers(self):
        """
        Load jawaban dari file answers.txt.
        
        Format file: nomor_soal,jawaban
        Contoh:
            1,8
            2,6
            3,12
        """
        if not os.path.exists(self.answers_file):
            raise FileNotFoundError(f"File '{self.answers_file}' tidak ditemukan!")
        
        answers_dict = {}
        
        with open(self.answers_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip baris kosong atau comment
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Parse: "1,8" -> question_id=1, answer=8
                    parts = line.split(',')
                    if len(parts) != 2:
                        raise ValueError(f"Format salah pada baris {line_num}")
                    
                    question_id = int(parts[0].strip())
                    answer = int(parts[1].strip())
                    
                    answers_dict[question_id] = answer
                
                except ValueError as e:
                    print(f"⚠ Warning: Tidak bisa parse baris {line_num}: {line} - {e}")
                    continue
        
        # Assign jawaban ke setiap question
        for question in self.questions:
            question_id = question['id']
            if question_id in answers_dict:
                question['correct_answer'] = answers_dict[question_id]
            else:
                print(f"⚠ Warning: Jawaban untuk soal #{question_id} tidak ditemukan!")
        
        print(f"✓ Loaded {len(answers_dict)} jawaban dari '{self.answers_file}'")
    
    def _validate_questions(self):
        """
        Validasi bahwa semua file yang dibutuhkan ada dan jawaban sudah di-set.
        """
        valid_questions = []
        
        for question in self.questions:
            is_valid = True
            question_id = question['id']
            
            # Cek file gambar ada
            if not os.path.exists(question['image_path']):
                print(f"⚠ Warning: Gambar soal #{question_id} tidak ditemukan: {question['image_path']}")
                is_valid = False
            
            # Cek file audio ada
            if not os.path.exists(question['audio_path']):
                print(f"⚠ Warning: Audio soal #{question_id} tidak ditemukan: {question['audio_path']}")
                is_valid = False
            
            # Cek jawaban sudah di-set
            if question['correct_answer'] is None:
                print(f"⚠ Warning: Jawaban soal #{question_id} tidak tersedia!")
                is_valid = False
            
            if is_valid:
                valid_questions.append(question)
        
        self.questions = valid_questions
        
        if len(self.questions) == 0:
            raise ValueError("Tidak ada soal yang valid! Periksa file data dan answers.txt")
        
        print(f"✓ Validasi selesai: {len(self.questions)} soal valid")
    
    def _randomize_questions(self):
        """
        Randomisasi urutan soal agar tidak monoton setiap run.
        """
        random.shuffle(self.questions)
        print("✓ Urutan soal telah di-randomize")
    
    def get_current_question(self) -> Optional[Dict]:
        """
        Mendapatkan soal yang sedang aktif.
        
        Returns:
            Dict: Question object dengan keys: id, image_path, audio_path, correct_answer
            None: Jika quiz sudah selesai
        """
        if self.is_finished():
            return None
        
        return self.questions[self.current_index]
    
    def check_answer(self, user_answer: int) -> bool:
        """
        Mengecek apakah jawaban user benar.
        
        Args:
            user_answer (int): Jawaban yang diberikan user (0-9)
        
        Returns:
            bool: True jika benar, False jika salah
        """
        current_question = self.get_current_question()
        
        if current_question is None:
            return False
        
        is_correct = (user_answer == current_question['correct_answer'])
        
        # Update statistics
        self.total_answered += 1
        if is_correct:
            self.correct_count += 1
        
        return is_correct
    
    def next_question(self) -> Optional[Dict]:
        """
        Pindah ke soal berikutnya.
        
        Returns:
            Dict: Question object berikutnya
            None: Jika sudah tidak ada soal lagi (quiz selesai)
        """
        self.current_index += 1
        return self.get_current_question()
    
    def get_score(self) -> Tuple[int, int, float]:
        """
        Mendapatkan skor saat ini.
        
        Returns:
            Tuple[int, int, float]: (correct_count, total_answered, percentage)
        """
        if self.total_answered == 0:
            percentage = 0.0
        else:
            percentage = (self.correct_count / self.total_answered) * 100
        
        return self.correct_count, self.total_answered, percentage
    
    def get_total_questions(self) -> int:
        """
        Mendapatkan total jumlah soal.
        
        Returns:
            int: Total soal yang tersedia
        """
        return len(self.questions)
    
    def is_finished(self) -> bool:
        """
        Mengecek apakah quiz sudah selesai (semua soal sudah dijawab).
        
        Returns:
            bool: True jika quiz selesai, False jika masih ada soal
        """
        return self.current_index >= len(self.questions)
    
    def reset(self):
        """
        Reset quiz untuk mulai dari awal.
        Randomize ulang urutan soal.
        """
        self.current_index = 0
        self.correct_count = 0
        self.total_answered = 0
        self._randomize_questions()
        print("\n✓ Quiz di-reset! Urutan soal di-randomize ulang.")
    
    def get_progress(self) -> str:
        """
        Mendapatkan progress quiz dalam format string.
        
        Returns:
            str: Progress dalam format "Soal X/Y"
        """
        current = min(self.current_index + 1, len(self.questions))
        total = len(self.questions)
        return f"Soal {current}/{total}"


# Test function
if __name__ == "__main__":
    print("Testing Quiz Manager Module...")
    print("-" * 60)
    
    try:
        # Test initialization
        quiz = QuizManager()
        
        print("\n" + "=" * 60)
        print("QUIZ INFO")
        print("=" * 60)
        print(f"Total soal: {quiz.get_total_questions()}")
        print(f"Progress: {quiz.get_progress()}")
        
        # Test get current question
        current = quiz.get_current_question()
        if current:
            print(f"\nSoal saat ini:")
            print(f"  ID: {current['id']}")
            print(f"  Gambar: {current['image_path']}")
            print(f"  Audio: {current['audio_path']}")
            print(f"  Jawaban: {current['correct_answer']}")
        
        print("\n✓ Quiz Manager test: PASSED")
        
    except Exception as e:
        print(f"\n✗ Quiz Manager test: FAILED - {str(e)}")
