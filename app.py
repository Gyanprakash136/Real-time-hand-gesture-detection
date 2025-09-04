import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time

# Gesture Classifier Class
class GestureClassifier:
    def __init__(self):
        """Initialize gesture classifier with rule-based logic"""
        self.gesture_names = ['Open Palm', 'Fist', 'Peace Sign', 'Thumbs Up']
    
    def classify_gesture(self, landmarks):
        """
        Classify gesture based on hand landmarks
        
        Args:
            landmarks: List of [x, y] coordinates for hand landmarks
            
        Returns:
            dict: {'gesture': gesture_name, 'confidence': confidence_score}
        """
        if len(landmarks) != 21:
            return None
        
        # Calculate finger states
        fingers_up = self._get_fingers_up(landmarks)
        
        # Classify based on finger patterns
        gesture, confidence = self._classify_finger_pattern(fingers_up, landmarks)
        
        if gesture:
            return {
                'gesture': gesture,
                'confidence': confidence
            }
        
        return None
    
    def _get_fingers_up(self, landmarks):
        """
        Determine which fingers are up based on landmark positions
        
        Returns:
            list: [thumb, index, middle, ring, pinky] - 1 if up, 0 if down
        """
        fingers = []
        
        # Thumb (compare x-coordinates due to thumb orientation)
        if landmarks[4][0] > landmarks[3][0]:  # Thumb tip vs thumb joint
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (compare y-coordinates)
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_pips = [6, 10, 14, 18]  # Index, Middle, Ring, Pinky PIPs
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] < landmarks[pip][1]:  # Tip above PIP
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def _classify_finger_pattern(self, fingers_up, landmarks):
        """
        Classify gesture based on finger pattern
        
        Args:
            fingers_up: List indicating which fingers are up
            landmarks: Hand landmark coordinates
            
        Returns:
            tuple: (gesture_name, confidence)
        """
        thumb, index, middle, ring, pinky = fingers_up
        
        # Open Palm: All fingers up
        if sum(fingers_up) >= 4:
            return 'Open Palm', 0.9
        
        # Fist: All fingers down
        elif sum(fingers_up) == 0:
            return 'Fist', 0.9
        
        # Peace Sign: Index and middle up, others down
        elif fingers_up == [0, 1, 1, 0, 0]:
            return 'Peace Sign', 0.95
        
        # Thumbs Up: Only thumb up
        elif fingers_up == [1, 0, 0, 0, 0]:
            # Additional check for thumb orientation
            if self._is_thumbs_up_orientation(landmarks):
                return 'Thumbs Up', 0.9
        
        # Partial matches with lower confidence
        elif index and middle and not ring and not pinky:
            return 'Peace Sign', 0.7
        
        elif thumb and sum(fingers_up) == 1:
            return 'Thumbs Up', 0.7
        
        elif sum(fingers_up) >= 3:
            return 'Open Palm', 0.6
        
        elif sum(fingers_up) <= 1:
            return 'Fist', 0.6
        
        return None, 0
    
    def _is_thumbs_up_orientation(self, landmarks):
        """
        Check if thumb is in proper thumbs-up orientation
        
        Args:
            landmarks: Hand landmark coordinates
            
        Returns:
            bool: True if thumb is pointing up
        """
        # Check if thumb tip is above wrist
        thumb_tip = landmarks[4]
        wrist = landmarks[0]
        
        return thumb_tip[1] < wrist[1] - 30  # Thumb significantly above wrist

# Hand Gesture Detector Class
class HandGestureDetector:
    def __init__(self):
        """Initialize the hand gesture detector using MediaPipe"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.gesture_classifier = GestureClassifier()
    
    def detect_gesture(self, frame):
        """
        Detect hand gestures in the given frame
        
        Args:
            frame: Input image frame
            
        Returns:
            tuple: (processed_frame, gesture_info)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        gesture_info = None
        
        # Draw hand landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract landmark coordinates
                landmarks = self._extract_landmarks(hand_landmarks, frame.shape)
                
                # Classify gesture
                gesture_info = self.gesture_classifier.classify_gesture(landmarks)
                
                # Draw bounding box
                x_coords = [lm[0] for lm in landmarks]
                y_coords = [lm[1] for lm in landmarks]
                x_min, x_max = min(x_coords) - 20, max(x_coords) + 20
                y_min, y_max = min(y_coords) - 20, max(y_coords) + 20
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
                
                # Display gesture name on frame
                if gesture_info:
                    gesture_name = gesture_info['gesture']
                    confidence = gesture_info['confidence']
                    
                    # Background for text
                    text = f"{gesture_name} ({confidence:.1%})"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (x_min, y_min - 40), 
                                (x_min + text_size[0] + 10, y_min), (0, 0, 0), -1)
                    
                    # Text
                    cv2.putText(frame, text, (x_min + 5, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame, gesture_info
    
    def _extract_landmarks(self, hand_landmarks, frame_shape):
        """Extract normalized landmark coordinates"""
        landmarks = []
        h, w = frame_shape[:2]
        
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        
        return landmarks

# Main Streamlit Application
def main():
    st.set_page_config(
        page_title="Real-Time Hand Gesture Recognition",
        page_icon="👋",
        layout="wide"
    )
    
    st.title("Real-Time Hand Gesture Recognition System")
    st.markdown("**AI Intern Assessment - Adhip Sarda**")
    st.markdown("---")
    
    # Sidebar with information
    with st.sidebar:
        st.header("Gesture Guide")
        st.markdown("""
        **Supported Gestures:**
        - Open Palm
        - Fist  
        - Peace Sign (V-sign)
        - Thumbs Up
        """)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Click 'Start Detection' to begin")
        st.markdown("2. Show your hand to the camera")
        st.markdown("3. Perform gestures clearly")
        st.markdown("4. The detected gesture will appear on screen")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed")
        
        # Control buttons
        start_button = st.button("Start Detection", type="primary")
        stop_button = st.button("Stop Detection")
        
        # Placeholder for video feed
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
    with col2:
        st.subheader("Detection Status")
        gesture_display = st.empty()
        confidence_display = st.empty()
        fps_display = st.empty()
    
    # Initialize session state
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    
    if start_button:
        st.session_state.detection_active = True
    if stop_button:
        st.session_state.detection_active = False
    
    if st.session_state.detection_active:
        run_gesture_detection(video_placeholder, gesture_display, 
                            confidence_display, fps_display, status_placeholder)

def run_gesture_detection(video_placeholder, gesture_display, 
                         confidence_display, fps_display, status_placeholder):
    """Main function to run gesture detection"""
    
    # Initialize the gesture detector
    detector = HandGestureDetector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        status_placeholder.error("Unable to access webcam. Please check your camera permissions.")
        return
    
    status_placeholder.success("Camera initialized successfully!")
    
    # FPS calculation
    fps_counter = 0
    start_time = time.time()
    
    try:
        while st.session_state.detection_active:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Failed to capture frame from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect gestures
            processed_frame, gesture_info = detector.detect_gesture(frame)
            
            # Update displays
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            
            if gesture_info:
                gesture_name = gesture_info.get('gesture', 'No Gesture')
                confidence = gesture_info.get('confidence', 0)
                
                gesture_display.markdown(f"### **{gesture_name}**")
                confidence_display.metric("Confidence", f"{confidence:.1%}")
            else:
                gesture_display.markdown("### **No Gesture Detected**")
                confidence_display.metric("Confidence", "0%")
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 10 == 0:  # Update every 10 frames
                elapsed_time = time.time() - start_time
                current_fps = fps_counter / elapsed_time
                fps_display.metric("FPS", f"{current_fps:.1f}")
            
            # Small delay to prevent overwhelming the interface
            time.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        status_placeholder.error(f"Error during detection: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
