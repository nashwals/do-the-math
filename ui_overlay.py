"""
UI Overlay untuk DO THE MATH
Menangani rendering UI elements seperti ready button, notifications, dll
Extracted dan refactored dari gesture_tracking.py
"""

import cv2
import numpy as np


class UIOverlay:
    """
    Class untuk menangani rendering UI overlay
    """
    
    def __init__(self, width, height, notification_duration=30):
        """
        Initialize UIOverlay
        
        Args:
            width: Lebar screen
            height: Tinggi screen
            notification_duration: Durasi notifikasi dalam frames
        """
        self.width = width
        self.height = height
        self.notification_duration = notification_duration
        
        # Calculate scaling factors (base: 1280x720)
        self.scale_x = width / 1280.0
        self.scale_y = height / 720.0
        self.min_scale = min(self.scale_x, self.scale_y)
        
        # Notification state
        self.notification_text = ""
        self.notification_timer = 0
    
    def get_scaled_font(self, base_size):
        """Get scaled font size"""
        return max(0.4, base_size * self.min_scale)
    
    def get_scaled_thickness(self, base_thickness):
        """Get scaled thickness"""
        return max(1, int(base_thickness * self.min_scale))
    
    def display_ready_button(self, img, gesture_status):
        """
        Menampilkan status ready/drawing button di kanan tengah (lebih ke bawah).
        
        Args:
            img: Frame gambar
            gesture_status: Dict dari GestureTracking.handle_gesture()
        """
        # GESER KE BAWAH - dari 3% jadi 40% (tengah-kanan)
        button_x = self.width - int(self.width * 0.12)
        button_y = int(self.height * 0.40)  # Geser ke bawah (dari 0.03)
        button_w = int(self.width * 0.10)
        button_h = int(self.height * 0.06)
        
        if gesture_status['action'] == 'charging':
            # Progress charging
            progress = gesture_status.get('progress', 0)
            
            # Background
            overlay = img.copy()
            cv2.rectangle(overlay, (button_x, button_y), 
                        (button_x + button_w, button_y + button_h),
                        (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # Border
            cv2.rectangle(img, (button_x, button_y), 
                        (button_x + button_w, button_y + button_h),
                        (0, 255, 255), 2)
            
            # Progress fill
            fill_w = int(button_w * progress)
            cv2.rectangle(img, (button_x, button_y), 
                        (button_x + fill_w, button_y + button_h),
                        (0, 255, 0), -1)
            
            # Text - BOLD
            font_scale = self.get_scaled_font(0.6)
            thickness = self.get_scaled_thickness(3)  # Bold
            cv2.putText(img, "READY...", 
                       (button_x + int(button_w * 0.08), button_y + int(button_h * 0.65)),
                       cv2.FONT_HERSHEY_TRIPLEX, font_scale, (255, 255, 255), thickness)
        
        elif gesture_status['ready'] and gesture_status['action'] == 'draw':
            # Drawing active
            overlay = img.copy()
            cv2.rectangle(overlay, (button_x, button_y), 
                        (button_x + button_w, button_y + button_h),
                        (128, 0, 128), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            cv2.rectangle(img, (button_x, button_y), 
                        (button_x + button_w, button_y + button_h),
                        (255, 0, 255), 2)
            
            # Text - BOLD
            font_scale = self.get_scaled_font(0.6)
            thickness = self.get_scaled_thickness(3)  # Bold
            cv2.putText(img, "DRAWING", 
                       (button_x + int(button_w * 0.05), button_y + int(button_h * 0.65)),
                       cv2.FONT_HERSHEY_TRIPLEX, font_scale, (255, 255, 255), thickness)
    
    def display_notification(self, img):
        """
        Menampilkan notifikasi hasil recognition di pojok kanan bawah.
        
        Args:
            img: Frame gambar
        """
        if self.notification_timer > 0:
            text_x = self.width - int(self.width * 0.16)
            text_y = self.height - int(self.height * 0.03)
            
            # Tentukan warna
            if "ANGKA:" in self.notification_text:
                color = (0, 255, 0)
            elif "TIDAK JELAS" in self.notification_text or "NOT LOADED" in self.notification_text:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            
            # BOLD FONT
            font_scale = self.get_scaled_font(0.7)
            thickness = self.get_scaled_thickness(3)  # Bold
            
            cv2.putText(img, self.notification_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_TRIPLEX, font_scale, color, thickness)
            
            self.notification_timer -= 1
    
    def display_finger_status(self, img, fingers):
        """
        Menampilkan status jari yang terdeteksi di pojok kiri bawah.
        
        Args:
            img: Frame gambar
            fingers: List finger status dari hand detector
        """
        if fingers:
            finger_text = f"Jari: {fingers}"
            text_x = int(self.width * 0.02)
            text_y = self.height - int(self.height * 0.03)
            
            font_scale = self.get_scaled_font(0.6)
            thickness = self.get_scaled_thickness(2)
            
            cv2.putText(img, finger_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    
    def show_notification(self, text):
        """
        Set notification text dan reset timer
        
        Args:
            text: Text notifikasi yang akan ditampilkan
        """
        self.notification_text = text
        self.notification_timer = self.notification_duration
    
    def display_instructions(self, img, y_position=None):
        """
        Menampilkan instruksi penggunaan
        
        Args:
            img: Frame gambar
            y_position: Posisi Y untuk instruksi (default: pojok kiri atas)
        """
        instructions = [
            "INSTRUKSI:",
            "1 Jari (telunjuk) = Draw",
            "4 Jari (tanpa jempol) = Submit & Recognize",
            "5 Jari = Clear Canvas",
            "ketik 'q' = Quit"
        ]
        
        if y_position is None:
            y_offset = 30
        else:
            y_offset = y_position
        
        font_scale = self.get_scaled_font(0.5)
        thickness = self.get_scaled_thickness(2)
        
        for i, text in enumerate(instructions):
            cv2.putText(img, text, (10, y_offset + (i * 30)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)