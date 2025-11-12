import cv2
import time
import sys

from modules.hand_tracker import HandTracker, GestureType
from modules.air_drawer import AirDrawer


def main():
    print("\nGestures:")
    print("  1. Index finger only    -> DRAWING MODE")
    print("     (hold for 1.5s to activate brush)")
    print("  2. All 5 fingers open   -> ERASE canvas")
    print("  3. 4 fingers (no pinky) -> SUBMIT answer")
    print("\nPress 'q' to quit")
    print("=" * 50)
    print()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    
    # Get frame size
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read first frame")
        return
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Initialize modules
    tracker = HandTracker(brush_delay=1.5)
    drawer = AirDrawer(canvas_size=(h, w), brush_color=(0, 0, 255), brush_thickness=10)
    
    # Tracking variables
    last_erase_time = 0
    last_submit_time = 0
    action_cooldown = 1.0
    
    prev_finger_pos = None
    drawing_active = False
    
    print("[INFO] Camera started. Show your hand to begin!")
    print()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hands
            results = tracker.detect_hands(frame)
            
            current_time = time.time()
            
            # Process if hand detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    landmarks = hand_landmarks.landmark
                    
                    # Detect gesture
                    current_gesture = tracker.detect_gesture(landmarks)
                    stable_gesture = tracker.get_stable_gesture(current_gesture)
                    
                    # Update brush state
                    brush_active = tracker.update_brush_state(stable_gesture)
                    
                    # Get fingertip position
                    finger_pos = tracker.get_index_finger_tip(
                        landmarks, 
                        (frame.shape[0], frame.shape[1])
                    )
                    
                    # Handle DRAWING mode
                    if stable_gesture == GestureType.DRAWING:
                        progress = tracker.get_brush_timer_progress()
                        tracker.draw_brush_indicator(frame, finger_pos, progress)
                        
                        if brush_active:
                            # Start or continue drawing
                            if not drawing_active:
                                drawer.start_drawing(finger_pos)
                                drawing_active = True
                                print("[DRAW] Started drawing stroke")
                            else:
                                # Continue drawing
                                drawer.add_point(finger_pos)
                            
                            cv2.putText(frame, "Drawing...", 
                                      (finger_pos[0] + 25, finger_pos[1]), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            # Waiting for brush activation
                            if drawing_active:
                                drawer.end_drawing()
                                drawing_active = False
                                print("[DRAW] Ended drawing stroke")
                    
                    # Handle ERASE mode
                    elif stable_gesture == GestureType.ERASE:
                        if current_time - last_erase_time > action_cooldown:
                            drawer.clear_canvas()
                            cv2.putText(frame, "CANVAS ERASED!", 
                                      (frame.shape[1]//2 - 150, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
                            print("[ACTION] Canvas erased!")
                            last_erase_time = current_time
                        
                        if drawing_active:
                            drawer.end_drawing()
                            drawing_active = False
                    
                    # Handle SUBMIT mode
                    elif stable_gesture == GestureType.SUBMIT:
                        if current_time - last_submit_time > action_cooldown:
                            cv2.putText(frame, "ANSWER SUBMITTED!", 
                                      (frame.shape[1]//2 - 180, 100),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
                            
                            # Process digit
                            digit_img = drawer.get_preprocessed_digit()
                            if digit_img is not None:
                                print("[SUBMIT] Jawaban anda sudah terkirim")
                                print("[SUBMIT] Digit shape:", digit_img.shape)
                                
                                # Show preprocessed digit (for debugging)
                                digit_display = cv2.resize(digit_img, (140, 140), 
                                                          interpolation=cv2.INTER_NEAREST)
                                # Place in corner
                                y_offset = 10
                                x_offset = frame.shape[1] - 150
                                frame[y_offset:y_offset+140, x_offset:x_offset+140] = \
                                    cv2.cvtColor((digit_display * 255).astype('uint8'), 
                                                cv2.COLOR_GRAY2BGR)
                                cv2.rectangle(frame, (x_offset, y_offset), 
                                            (x_offset+140, y_offset+140), (255, 0, 255), 2)
                            else:
                                print("[WARNING] No drawing to submit")
                            
                            last_submit_time = current_time
                        
                        if drawing_active:
                            drawer.end_drawing()
                            drawing_active = False
                    
                    # Other gesture
                    else:
                        if drawing_active:
                            drawer.end_drawing()
                            drawing_active = False
                    
                    # Draw gesture info
                    tracker.draw_gesture_info(frame, stable_gesture)
                    
                    prev_finger_pos = finger_pos
            
            else:
                # No hand detected
                if drawing_active:
                    drawer.end_drawing()
                    drawing_active = False
            
            # Overlay drawing on frame
            frame = drawer.overlay_on_frame(frame, alpha=0.7)
            
            # Show drawing status
            if drawer.has_drawing():
                cv2.putText(frame, "Drawing exists", (10, frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("DO THE MATH - Hand Tracker + Air Drawer", frame)
            
            # Check quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.release()
        print("\n[INFO] Demo ended successfully")
        print("=" * 50)


if __name__ == "__main__":
    main()