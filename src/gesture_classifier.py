import numpy as np
import math

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
