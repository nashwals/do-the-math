"""
Audio Player Module untuk DO THE MATH!
======================================

Modul ini mengelola:
1. Inisialisasi pygame.mixer untuk audio playback
2. Play audio soal matematika
3. Play sound effects (applause, correct, wrong)
4. Stop dan cleanup audio

Menggunakan pygame.mixer untuk non-blocking audio playback.
"""

import pygame
import os
from typing import Optional


class AudioPlayer:
    """
    Class untuk mengelola semua audio playback dalam aplikasi.
    
    Attributes:
        initialized (bool): Status inisialisasi pygame.mixer
        current_audio (pygame.mixer.Sound): Audio yang sedang diputar
        sounds_folder (str): Path ke folder sound effects
        audio_queue (list): Queue untuk sequential audio playback
        is_playing_sequence (bool): Flag untuk sequence playback
    """
    
    def __init__(self, sounds_folder: str = "assets/audio/"):
        """
        Inisialisasi Audio Player.
        
        Args:
            sounds_folder (str): Path ke folder yang berisi sound effects
        """
        self.initialized = False
        self.current_audio: Optional[pygame.mixer.Sound] = None
        self.sounds_folder = sounds_folder
        self.audio_queue = []  # Queue untuk sequential audio
        self.is_playing_sequence = False  # Flag untuk sequence mode
        
        self._initialize_mixer()
    
    def _initialize_mixer(self):
        """
        Inisialisasi pygame.mixer untuk audio playback.
        
        Setup:
        - Frequency: 44100 Hz (CD quality)
        - Size: -16 (16-bit audio)
        - Channels: 2 (stereo)
        - Buffer: 512 (balance between latency and performance)
        """
        try:
            # Initialize pygame mixer
            pygame.mixer.init(
                frequency=44100,  # Sample rate
                size=-16,         # 16-bit audio
                channels=2,       # Stereo
                buffer=512        # Buffer size
            )
            
            self.initialized = True
            print("✓ Audio Player initialized successfully")
            
        except Exception as e:
            print(f"⚠ Warning: Gagal inisialisasi audio - {str(e)}")
            print("  Program akan berjalan tanpa audio")
            self.initialized = False
    
    def play_question_audio(self, audio_path: str) -> bool:
        """
        Play audio soal matematika.
        
        Audio akan diputar secara non-blocking (program tetap jalan).
        Jika ada audio yang sedang diputar, akan di-stop terlebih dahulu.
        
        Args:
            audio_path (str): Path ke file audio soal (.wav)
        
        Returns:
            bool: True jika berhasil play, False jika gagal
        """
        if not self.initialized:
            print("⚠ Audio player tidak terinisialisasi")
            return False
        
        # Cek apakah file ada
        if not os.path.exists(audio_path):
            print(f"⚠ Warning: Audio file tidak ditemukan: {audio_path}")
            return False
        
        try:
            # Stop audio sebelumnya jika ada
            self.stop_audio()
            
            # Load dan play audio baru
            self.current_audio = pygame.mixer.Sound(audio_path)
            self.current_audio.play()
            
            print(f"♪ Playing audio: {os.path.basename(audio_path)}")
            return True
            
        except Exception as e:
            print(f"⚠ Error playing audio: {str(e)}")
            return False
    
    def play_sound_effect(self, effect_name: str, volume: float = 1.0) -> bool:
        """
        Play sound effect (applause, correct, wrong, dll).
        
        Sound effects ada di folder sounds/.
        
        Args:
            effect_name (str): Nama sound effect tanpa ekstensi (e.g., "applause")
            volume (float): Volume sound effect (0.0 - 1.0)
        
        Returns:
            bool: True jika berhasil play, False jika gagal
        """
        if not self.initialized:
            return False
        
        # Construct path ke sound effect
        sound_path = os.path.join(self.sounds_folder, f"{effect_name}.wav")
        
        # Cek apakah file ada
        if not os.path.exists(sound_path):
            print(f"⚠ Sound effect tidak ditemukan: {sound_path}")
            print(f"  Silakan tambahkan file '{effect_name}.wav' ke folder '{self.sounds_folder}/'")
            return False
        
        try:
            # Load dan play sound effect
            sound = pygame.mixer.Sound(sound_path)
            sound.set_volume(volume)
            sound.play()
            
            print(f"♪ Playing sound effect: {effect_name}")
            return True
            
        except Exception as e:
            print(f"⚠ Error playing sound effect: {str(e)}")
            return False
    
    def play_applause(self) -> bool:
        """
        Shortcut untuk play applause sound effect.
        
        Returns:
            bool: True jika berhasil play, False jika gagal
        """
        return self.play_sound_effect("applause", volume=0.8)
    
    def play_correct_sound(self) -> bool:
        """
        Shortcut untuk play correct answer sound effect (opsional).
        
        Returns:
            bool: True jika berhasil play, False jika gagal
        """
        return self.play_sound_effect("correct", volume=0.7)
    
    def play_wrong_sound(self) -> bool:
        """
        Shortcut untuk play wrong answer sound effect (opsional).
        
        Returns:
            bool: True jika berhasil play, False jika gagal
        """
        return self.play_sound_effect("wrong", volume=0.6)
    
    def play_correct_sequence(self) -> bool:
        """
        Play sequential audio untuk jawaban benar: correct -> applause.
        
        Audio akan diputar secara sequential (tidak parallel).
        Menggunakan audio_queue untuk manage sequence.
        
        Returns:
            bool: True jika berhasil start sequence, False jika gagal
        """
        if not self.initialized:
            return False
        
        # Build path untuk sound effects
        correct_path = os.path.join(self.sounds_folder, "correct.wav")
        applause_path = os.path.join(self.sounds_folder, "applause.wav")
        
        # Validasi file ada
        if not os.path.exists(correct_path):
            print(f"⚠ Sound effect tidak ditemukan: {correct_path}")
            return False
        
        if not os.path.exists(applause_path):
            print(f"⚠ Sound effect tidak ditemukan: {applause_path}")
            return False
        
        try:
            # Stop audio sebelumnya
            self.stop_audio()
            
            # Setup audio queue untuk sequential playback
            self.audio_queue = [
                ("correct", correct_path, 0.7),   # (name, path, volume)
                ("applause", applause_path, 0.8)
            ]
            
            self.is_playing_sequence = True
            
            # Play first audio in sequence
            self._play_next_in_sequence()
            
            print("♪ Starting audio sequence: correct -> applause")
            return True
            
        except Exception as e:
            print(f"⚠ Error starting audio sequence: {str(e)}")
            self.is_playing_sequence = False
            self.audio_queue = []
            return False
    
    def _play_next_in_sequence(self):
        """
        Internal method untuk play audio berikutnya dalam sequence.
        
        Dipanggil secara internal untuk manage sequential playback.
        """
        if not self.audio_queue or not self.is_playing_sequence:
            self.is_playing_sequence = False
            return
        
        # Get next audio dari queue
        name, path, volume = self.audio_queue.pop(0)
        
        try:
            # Load dan play audio
            self.current_audio = pygame.mixer.Sound(path)
            self.current_audio.set_volume(volume)
            self.current_audio.play()
            
            print(f"♪ Playing in sequence: {name}")
            
        except Exception as e:
            print(f"⚠ Error playing {name}: {str(e)}")
            self.is_playing_sequence = False
            self.audio_queue = []
    
    def update_sequence(self):
        """
        Update sequential audio playback.
        
        Method ini harus dipanggil setiap frame dalam main loop
        untuk manage sequential audio playback.
        
        Ketika audio saat ini selesai, otomatis play audio berikutnya.
        """
        if not self.is_playing_sequence:
            return
        
        # Check apakah audio saat ini masih playing
        if not pygame.mixer.get_busy():
            # Audio selesai, play next in sequence
            if len(self.audio_queue) > 0:
                self._play_next_in_sequence()
            else:
                # Sequence selesai
                self.is_playing_sequence = False
                print("♪ Audio sequence completed")

    
    def stop_audio(self):
        """
        Stop audio yang sedang diputar.
        
        Berguna untuk stop audio soal saat user submit jawaban.
        """
        if not self.initialized:
            return
        
        try:
            # Stop semua channel yang sedang play
            pygame.mixer.stop()
            self.current_audio = None
            
        except Exception as e:
            print(f"⚠ Error stopping audio: {str(e)}")
    
    def is_playing(self) -> bool:
        """
        Cek apakah ada audio yang sedang diputar.
        
        Returns:
            bool: True jika ada audio yang play, False jika tidak
        """
        if not self.initialized:
            return False
        
        return pygame.mixer.get_busy()
    
    def is_sequence_playing(self) -> bool:
        """
        Cek apakah sequential audio masih playing.
        
        Returns:
            bool: True jika sequence masih berjalan, False jika tidak
        """
        return self.is_playing_sequence
    
    def set_volume(self, volume: float):
        """
        Set volume global untuk semua audio.
        
        Args:
            volume (float): Volume level (0.0 - 1.0)
        """
        if not self.initialized:
            return
        
        try:
            volume = max(0.0, min(1.0, volume))  # Clamp 0-1
            pygame.mixer.music.set_volume(volume)
            print(f"♪ Volume set to {int(volume * 100)}%")
            
        except Exception as e:
            print(f"⚠ Error setting volume: {str(e)}")
    
    def cleanup(self):
        """
        Cleanup audio player saat program exit.
        
        Harus dipanggil sebelum program close untuk proper cleanup.
        """
        if not self.initialized:
            return
        
        try:
            self.stop_audio()
            pygame.mixer.quit()
            print("✓ Audio Player cleaned up")
            
        except Exception as e:
            print(f"⚠ Error during cleanup: {str(e)}")


# Test function
if __name__ == "__main__":
    print("Testing Audio Player Module...")
    print("-" * 60)
    
    try:
        # Test initialization
        player = AudioPlayer()
        
        if player.initialized:
            print("\n✓ Audio Player test: INITIALIZED")
            
            # Test sound effect
            print("\nTesting sound effects...")
            print("Note: Pastikan file 'assets/audio/applause.wav' ada untuk test ini")
            
            # Cek folder sounds
            if not os.path.exists("assets/audio/"):
                print("⚠ Folder 'assets/audio/' tidak ditemukan")
                print("  Membuat folder 'assets/audio/'...")
                os.makedirs("sounds", exist_ok=True)
            
            # Test play applause
            player.play_applause()
            
            print("\n✓ Audio Player test: PASSED")
        else:
            print("\n⚠ Audio Player test: INITIALIZED (without audio support)")
        
        # Cleanup
        player.cleanup()
        
    except Exception as e:
        print(f"\n✗ Audio Player test: FAILED - {str(e)}")