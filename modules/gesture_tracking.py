import cv2
import mediapipe as mp
import numpy as np
import time

class CompactGestureTracker:
    def __init__(self, brush_delay=1.5):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        
        # Canvas & Drawing
        self.canvas = None
        self.prev_point = None
        self.brush_color = (0, 0, 255)  # Red
        self.brush_thickness = 10
        
        # Timing & State
        self.brush_delay = brush_delay
        self.brush_timer = None
        self.gesture_buffer = []
        
        print(f"[Tracker] Ready! Brush delay: {brush_delay}s")
    
    def init_canvas(self, h, w):
        """Initialize canvas"""
        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    def process_frame(self, frame):
        """Main processing: detect hands and return results"""
        return self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    def get_gesture(self, landmarks):
        """Deteksi gesture dari landmarks"""
        # Get finger tips and pips
        tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]
        pips = [landmarks[i] for i in [3, 6, 10, 14, 18]]
        wrist = landmarks[0]
        
        # Check which fingers are up
        threshold = 0.02
        thumb = (abs(tips[0].x - wrist.x) + abs(tips[0].y - wrist.y)) > \
                (abs(pips[0].x - wrist.x) + abs(pips[0].y - wrist.y))
        fingers = [thumb] + [(pips[i].y - tips[i].y) > threshold for i in range(1, 5)]
        
        count = sum(fingers)
        
        # Gesture logic
        if count >= 5:
            return "ERASE"
        if count >= 4 and not fingers[4]:  # No pinky
            return "SUBMIT"
        if fingers[1] and not any(fingers[2:]):  # Only index
            return "DRAWING"
        return "NONE"
    
    def stabilize_gesture(self, gesture):
        """Stabilkan gesture dengan buffer"""
        self.gesture_buffer.append(gesture)
        if len(self.gesture_buffer) > 5:
            self.gesture_buffer.pop(0)
        
        if len(self.gesture_buffer) < 3:
            return "NONE"
        
        # Return most common gesture
        counts = {g: self.gesture_buffer.count(g) for g in set(self.gesture_buffer)}
        best = max(counts, key=counts.get)
        return best if counts[best] >= len(self.gesture_buffer) * 0.5 else "NONE"
    
    def update_brush(self, gesture):
        """Update brush state berdasarkan gesture"""
        if gesture == "DRAWING":
            if not self.brush_timer:
                self.brush_timer = time.time()
            return (time.time() - self.brush_timer) >= self.brush_delay
        else:
            self.brush_timer = None
            return False
    
    def get_progress(self):
        """Get brush activation progress (0.0-1.0)"""
        if not self.brush_timer:
            return 0.0
        return min((time.time() - self.brush_timer) / self.brush_delay, 1.0)
    
    def draw(self, point):
        """Draw line on canvas"""
        if self.prev_point:
            cv2.line(self.canvas, self.prev_point, point, 
                    self.brush_color, self.brush_thickness, cv2.LINE_AA)
        self.prev_point = point
    
    def clear(self):
        """Clear canvas"""
        self.canvas = np.zeros_like(self.canvas)
        self.prev_point = None
    
    def overlay(self, frame):
        """Overlay canvas on frame"""
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        if np.any(mask):
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
            result = frame.copy()
            alpha = 0.7
            drawing = mask_3ch > 0
            result[drawing] = (frame[drawing] * (1-alpha) + 
                              self.canvas[drawing] * alpha).astype(np.uint8)
            return result
        return frame
    
    def draw_ui(self, frame, gesture, finger_pos=None):
        """Draw all UI elements"""
        h, w = frame.shape[:2]
        
        # Gesture info
        colors = {"NONE": (200,200,200), "DRAWING": (0,255,0), 
                 "ERASE": (0,165,255), "SUBMIT": (255,0,255)}
        texts = {"NONE": "No Gesture", "DRAWING": "DRAWING MODE",
                "ERASE": "ERASE MODE", "SUBMIT": "SUBMIT ANSWER"}
        
        text = texts.get(gesture, "Unknown")
        color = colors.get(gesture, (255,255,255))
        
        # Background box
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (5, 5), (tw+15, th+15), (0,0,0), -1)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2, cv2.LINE_AA)
        
        # Brush indicator
        if finger_pos and gesture == "DRAWING":
            x, y = finger_pos
            progress = self.get_progress()
            
            if progress < 1.0:
                # Progress ring
                cv2.circle(frame, (x,y), 20, (200,200,200), 2)
                angle = int(360 * progress)
                if angle > 0:
                    cv2.ellipse(frame, (x,y), (20,20), -90, 0, angle, (0,255,255), 3)
                cv2.circle(frame, (x,y), 5, (0,255,255), -1)
            else:
                # Active brush
                cv2.circle(frame, (x,y), 15, (0,0,255), -1)
                cv2.circle(frame, (x,y), 18, (0,0,255), 2)


def main():
    """Main demo"""
    print("="*60)
    print("  DO THE MATH! - COMPACT DEMO")
    print("="*60)
    print("\n☝️  Index finger → DRAWING (hold 1.5s)")
    print("🖐️  5 fingers → ERASE")
    print("🖖  4 fingers → SUBMIT")
    print("\nPress 'q' to quit\n")
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Camera error!")
        return
    
    # Get frame size
    ret, frame = cap.read()
    if not ret:
        return
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Initialize tracker
    tracker = CompactGestureTracker(brush_delay=1.5)
    tracker.init_canvas(h, w)
    
    # Action cooldowns
    last_action = {"ERASE": 0, "SUBMIT": 0}
    cooldown = 1.0
    
    print("✅ Ready!\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            results = tracker.process_frame(frame)
            current_time = time.time()
            
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    # Draw skeleton
                    tracker.mp_draw.draw_landmarks(
                        frame, hand, tracker.mp_hands.HAND_CONNECTIONS)
                    
                    # Get gesture
                    gesture = tracker.get_gesture(hand.landmark)
                    stable = tracker.stabilize_gesture(gesture)
                    brush_active = tracker.update_brush(stable)
                    
                    # Get finger position
                    tip = hand.landmark[8]
                    pos = (int(tip.x * w), int(tip.y * h))
                    
                    # Handle gesture
                    if stable == "DRAWING":
                        if brush_active:
                            tracker.draw(pos)
                        else:
                            tracker.prev_point = None
                    
                    elif stable == "ERASE" and current_time - last_action["ERASE"] > cooldown:
                        tracker.clear()
                        print("[ACTION] Canvas erased!")
                        last_action["ERASE"] = current_time
                        tracker.prev_point = None
                    
                    elif stable == "SUBMIT" and current_time - last_action["SUBMIT"] > cooldown:
                        print("[ACTION] Answer submitted!")
                        last_action["SUBMIT"] = current_time
                        tracker.prev_point = None
                    
                    else:
                        tracker.prev_point = None
                    
                    # Draw UI
                    tracker.draw_ui(frame, stable, pos if stable == "DRAWING" else None)
            else:
                tracker.prev_point = None
            
            # Overlay and show
            frame = tracker.overlay(frame)
            cv2.imshow("DO THE MATH! - Compact", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.hands.close()
        print("✅ Done!\n")


if __name__ == "__main__":
    main()