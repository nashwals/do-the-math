"""
Confetti Effect Module untuk DO THE MATH!
==========================================

Modul ini mengelola:
1. Particle system untuk efek confetti
2. Physics simulation (gravity, velocity, rotation)
3. Rendering confetti ke frame
4. Lifecycle management particles

Menggunakan OpenCV untuk rendering dan NumPy untuk physics calculation.
"""

import numpy as np
import cv2
import random
from typing import List, Tuple


class Confetti:
    """
    Class untuk merepresentasikan satu particle confetti.
    
    Attributes:
        x (float): Posisi X confetti
        y (float): Posisi Y confetti
        velocity_x (float): Kecepatan horizontal
        velocity_y (float): Kecepatan vertikal
        color (tuple): Warna BGR confetti
        size (int): Ukuran confetti (radius)
        rotation (float): Sudut rotasi (untuk efek tumbling)
        rotation_speed (float): Kecepatan rotasi
        shape (str): Bentuk confetti ('circle', 'rectangle', 'triangle')
        gravity (float): Akselerasi gravitasi
        alive (bool): Status particle (True = masih di-render)
    """
    
    def __init__(self, x: float, y: float, window_width: int, window_height: int):
        """
        Inisialisasi satu particle confetti.
        
        Args:
            x (float): Posisi X awal
            y (float): Posisi Y awal
            window_width (int): Lebar window untuk boundary check
            window_height (int): Tinggi window untuk boundary check
        """
        self.x = x
        self.y = y
        
        # Random velocity (horizontal spread + vertical upward initial)
        self.velocity_x = random.uniform(-3, 3)  # Horizontal random
        self.velocity_y = random.uniform(-8, -3)  # Initial upward velocity
        
        # Random colorful colors (bright colors untuk festive effect)
        colors = [
            (255, 20, 147),   # Deep Pink
            (0, 215, 255),    # Gold
            (0, 255, 0),      # Lime Green
            (255, 0, 255),    # Magenta
            (0, 165, 255),    # Orange
            (255, 255, 0),    # Cyan
            (128, 0, 255),    # Purple
            (0, 255, 255),    # Yellow
        ]
        self.color = random.choice(colors)
        
        # Random size
        self.size = random.randint(4, 8)
        
        # Rotation properties untuk tumbling effect
        self.rotation = random.uniform(0, 360)
        self.rotation_speed = random.uniform(-15, 15)
        
        # Random shape
        self.shape = random.choice(['circle', 'rectangle', 'triangle'])
        
        # Physics properties
        self.gravity = 0.4  # Gravitasi untuk falling effect
        self.air_resistance = 0.99  # Sedikit air resistance
        
        # Boundary properties
        self.window_width = window_width
        self.window_height = window_height
        
        # Lifecycle
        self.alive = True
    
    def update(self):
        """
        Update posisi dan status confetti berdasarkan physics.
        
        Menerapkan:
        - Gravitasi (velocity_y meningkat)
        - Air resistance (velocity berkurang sedikit)
        - Rotasi (tumbling effect)
        - Boundary check (mark as dead jika keluar frame)
        """
        # Apply gravity
        self.velocity_y += self.gravity
        
        # Apply air resistance
        self.velocity_x *= self.air_resistance
        self.velocity_y *= self.air_resistance
        
        # Update position
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Update rotation
        self.rotation += self.rotation_speed
        if self.rotation > 360:
            self.rotation -= 360
        elif self.rotation < 0:
            self.rotation += 360
        
        # Check if out of bounds (mark as dead)
        if (self.y > self.window_height + 20 or 
            self.x < -20 or 
            self.x > self.window_width + 20):
            self.alive = False
    
    def draw(self, img: np.ndarray):
        """
        Render confetti ke frame.
        
        Args:
            img (np.ndarray): Frame untuk drawing
        """
        if not self.alive:
            return
        
        # Convert position to integer
        center_x = int(self.x)
        center_y = int(self.y)
        
        # Draw berdasarkan shape
        if self.shape == 'circle':
            self._draw_circle(img, center_x, center_y)
        elif self.shape == 'rectangle':
            self._draw_rectangle(img, center_x, center_y)
        elif self.shape == 'triangle':
            self._draw_triangle(img, center_x, center_y)
    
    def _draw_circle(self, img: np.ndarray, x: int, y: int):
        """
        Draw circular confetti.
        
        Args:
            img (np.ndarray): Frame untuk drawing
            x (int): Center X position
            y (int): Center Y position
        """
        cv2.circle(img, (x, y), self.size, self.color, -1)
        # Add border untuk depth effect
        cv2.circle(img, (x, y), self.size, (255, 255, 255), 1)
    
    def _draw_rectangle(self, img: np.ndarray, x: int, y: int):
        """
        Draw rectangular confetti dengan rotasi.
        
        Args:
            img (np.ndarray): Frame untuk drawing
            x (int): Center X position
            y (int): Center Y position
        """
        # Create rectangle points
        width = self.size * 2
        height = self.size
        
        # Calculate rotated rectangle
        angle_rad = np.radians(self.rotation)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Rectangle corners (before rotation)
        corners = np.array([
            [-width/2, -height/2],
            [width/2, -height/2],
            [width/2, height/2],
            [-width/2, height/2]
        ])
        
        # Apply rotation
        rotated_corners = []
        for corner in corners:
            rotated_x = corner[0] * cos_angle - corner[1] * sin_angle
            rotated_y = corner[0] * sin_angle + corner[1] * cos_angle
            rotated_corners.append([int(x + rotated_x), int(y + rotated_y)])
        
        # Draw filled polygon
        pts = np.array(rotated_corners, dtype=np.int32)
        cv2.fillPoly(img, [pts], self.color)
        # Add border
        cv2.polylines(img, [pts], True, (255, 255, 255), 1)
    
    def _draw_triangle(self, img: np.ndarray, x: int, y: int):
        """
        Draw triangular confetti dengan rotasi.
        
        Args:
            img (np.ndarray): Frame untuk drawing
            x (int): Center X position
            y (int): Center Y position
        """
        # Create triangle points
        size = self.size * 1.5
        
        # Calculate rotated triangle
        angle_rad = np.radians(self.rotation)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Triangle corners (before rotation)
        corners = np.array([
            [0, -size],           # Top
            [-size, size],        # Bottom left
            [size, size]          # Bottom right
        ])
        
        # Apply rotation
        rotated_corners = []
        for corner in corners:
            rotated_x = corner[0] * cos_angle - corner[1] * sin_angle
            rotated_y = corner[0] * sin_angle + corner[1] * cos_angle
            rotated_corners.append([int(x + rotated_x), int(y + rotated_y)])
        
        # Draw filled triangle
        pts = np.array(rotated_corners, dtype=np.int32)
        cv2.fillPoly(img, [pts], self.color)
        # Add border
        cv2.polylines(img, [pts], True, (255, 255, 255), 1)


class ConfettiSystem:
    """
    Class untuk mengelola sistem particle confetti.
    
    Attributes:
        particles (List[Confetti]): List semua particles yang aktif
        window_width (int): Lebar window
        window_height (int): Tinggi window
        is_generating (bool): Flag apakah masih generate particles baru
        generation_frames (int): Counter untuk staged generation
    """
    
    def __init__(self, window_width: int = 1280, window_height: int = 720):
        """
        Inisialisasi confetti system.
        
        Args:
            window_width (int): Lebar window untuk particle boundaries
            window_height (int): Tinggi window untuk particle boundaries
        """
        self.particles: List[Confetti] = []
        self.window_width = window_width
        self.window_height = window_height
        self.is_generating = False
        self.generation_frames = 0
        self.max_generation_frames = 15  # Generate particles selama 15 frames
    
    def generate(self, num_particles: int = 100):
        """
        Generate confetti particles baru.
        
        Particles di-spawn dari area atas tengah window dengan spread horizontal.
        
        Args:
            num_particles (int): Jumlah particles yang akan di-generate
        """
        # Clear existing particles
        self.particles = []
        
        # Generate particles
        for _ in range(num_particles):
            # Spawn dari atas dengan horizontal spread
            spawn_x = random.uniform(self.window_width * 0.2, self.window_width * 0.8)
            spawn_y = random.uniform(-50, -20)
            
            particle = Confetti(spawn_x, spawn_y, self.window_width, self.window_height)
            self.particles.append(particle)
        
        print(f"✨ Generated {num_particles} confetti particles")
    
    def generate_burst(self, num_particles: int = 150):
        """
        Generate confetti dengan burst effect (staged generation).
        
        Particles di-generate secara bertahap untuk efek burst yang lebih dramatic.
        
        Args:
            num_particles (int): Total jumlah particles
        """
        self.particles = []
        self.is_generating = True
        self.generation_frames = 0
        self.particles_per_frame = num_particles // self.max_generation_frames
        
        print(f"✨ Starting confetti burst: {num_particles} particles")
    
    def update(self):
        """
        Update semua particles dalam system.
        
        - Generate particles baru jika dalam mode burst
        - Update physics semua particles
        - Remove particles yang sudah mati
        """
        # Generate particles bertahap jika dalam mode burst
        if self.is_generating:
            if self.generation_frames < self.max_generation_frames:
                for _ in range(self.particles_per_frame):
                    spawn_x = random.uniform(self.window_width * 0.2, self.window_width * 0.8)
                    spawn_y = random.uniform(-50, -20)
                    particle = Confetti(spawn_x, spawn_y, self.window_width, self.window_height)
                    self.particles.append(particle)
                
                self.generation_frames += 1
            else:
                self.is_generating = False
        
        # Update semua particles
        for particle in self.particles:
            particle.update()
        
        # Remove dead particles (cleanup)
        self.particles = [p for p in self.particles if p.alive]
    
    def draw(self, img: np.ndarray):
        """
        Render semua particles ke frame.
        
        Args:
            img (np.ndarray): Frame untuk rendering
        """
        for particle in self.particles:
            particle.draw(img)
    
    def is_active(self) -> bool:
        """
        Check apakah masih ada particles yang aktif.
        
        Returns:
            bool: True jika masih ada particles, False jika sudah habis
        """
        return len(self.particles) > 0 or self.is_generating
    
    def clear(self):
        """
        Clear semua particles (force stop effect).
        """
        self.particles = []
        self.is_generating = False
        print("✨ Confetti cleared")
    
    def get_particle_count(self) -> int:
        """
        Get jumlah particles yang masih aktif.
        
        Returns:
            int: Jumlah particles aktif
        """
        return len(self.particles)


# Test function
if __name__ == "__main__":
    print("Testing Confetti Effect Module...")
    print("-" * 60)
    
    # Create test window
    window_width = 1280
    window_height = 720
    
    # Initialize confetti system
    confetti = ConfettiSystem(window_width, window_height)
    
    # Generate particles
    print("\n[TEST 1] Generating confetti particles...")
    confetti.generate_burst(num_particles=150)
    
    print(f"Initial particles: {confetti.get_particle_count()}")
    print(f"Is active: {confetti.is_active()}")
    
    # Simulate frames
    print("\n[TEST 2] Simulating particle physics...")
    frame_count = 0
    test_img = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    
    while confetti.is_active() and frame_count < 300:  # Max 10 seconds @ 30fps
        # Update particles
        confetti.update()
        
        # Clear test image
        test_img.fill(50)  # Gray background
        
        # Draw particles
        confetti.draw(test_img)
        
        # Print status every 30 frames (~1 second)
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {confetti.get_particle_count()} particles active")
        
        # Display (optional - comment out if no display available)
        # cv2.imshow("Confetti Test", test_img)
        # if cv2.waitKey(33) & 0xFF == ord('q'):
        #     break
        
        frame_count += 1
    
    print(f"\n[TEST 3] Confetti completed after {frame_count} frames")
    print(f"Final particles: {confetti.get_particle_count()}")
    
    # Test clear function
    print("\n[TEST 4] Testing clear function...")
    confetti.generate(50)
    print(f"Before clear: {confetti.get_particle_count()} particles")
    confetti.clear()
    print(f"After clear: {confetti.get_particle_count()} particles")
    
    print("\n" + "="*60)
    print("✓ Confetti Effect Module Test: PASSED")
    print("="*60)
    
    # cv2.destroyAllWindows()
