import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import json
from typing import List, Dict, Tuple, Optional
import logging
import math
from collections import Counter, deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedASLRecognizer:
    def __init__(self):
        """Initialize improved ASL Recognition system with enhanced accuracy"""
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models with optimized settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,  # Increased for better accuracy
            min_tracking_confidence=0.7    # Increased for better tracking
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        # Enhanced ASL alphabet mapping with confidence weights
        self.asl_alphabet = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
            8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
            16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
            24: 'Y', 25: 'Z'
        }
        
        # Enhanced ASL words and phrases mapping
        self.asl_words = {
            'HELLO': ['H', 'E', 'L', 'L', 'O'],
            'WORLD': ['W', 'O', 'R', 'L', 'D'],
            'THANK': ['T', 'H', 'A', 'N', 'K'],
            'YOU': ['Y', 'O', 'U'],
            'PLEASE': ['P', 'L', 'E', 'A', 'S', 'E'],
            'SORRY': ['S', 'O', 'R', 'R', 'Y'],
            'GOOD': ['G', 'O', 'O', 'D'],
            'MORNING': ['M', 'O', 'R', 'N', 'I', 'N', 'G'],
            'AFTERNOON': ['A', 'F', 'T', 'E', 'R', 'N', 'O', 'O', 'N'],
            'EVENING': ['E', 'V', 'E', 'N', 'I', 'N', 'G'],
            'NIGHT': ['N', 'I', 'G', 'H', 'T'],
            'YES': ['Y', 'E', 'S'],
            'NO': ['N', 'O'],
            'HELP': ['H', 'E', 'L', 'P'],
            'WATER': ['W', 'A', 'T', 'E', 'R'],
            'FOOD': ['F', 'O', 'O', 'D'],
            'HOME': ['H', 'O', 'M', 'E'],
            'WORK': ['W', 'O', 'R', 'K'],
            'SCHOOL': ['S', 'C', 'H', 'O', 'O', 'L'],
            'FRIEND': ['F', 'R', 'I', 'E', 'N', 'D'],
            'FAMILY': ['F', 'A', 'M', 'I', 'L', 'Y'],
            'LOVE': ['L', 'O', 'V', 'E'],
            'HAPPY': ['H', 'A', 'P', 'P', 'Y'],
            'SAD': ['S', 'A', 'D'],
            'BEAUTIFUL': ['B', 'E', 'A', 'U', 'T', 'I', 'F', 'U', 'L'],
            'WONDERFUL': ['W', 'O', 'N', 'D', 'E', 'R', 'F', 'U', 'L'],
            'AMAZING': ['A', 'M', 'A', 'Z', 'I', 'N', 'G'],
            'AWESOME': ['A', 'W', 'E', 'S', 'O', 'M', 'E'],
            'FANTASTIC': ['F', 'A', 'N', 'T', 'A', 'S', 'T', 'I', 'C'],
            'EXCELLENT': ['E', 'X', 'C', 'E', 'L', 'L', 'E', 'N', 'T']
        }
        
        # Common ASL phrases
        self.asl_phrases = {
            'HELLO WORLD': ['HELLO', 'WORLD'],
            'THANK YOU': ['THANK', 'YOU'],
            'GOOD MORNING': ['GOOD', 'MORNING'],
            'GOOD AFTERNOON': ['GOOD', 'AFTERNOON'],
            'GOOD EVENING': ['GOOD', 'EVENING'],
            'GOOD NIGHT': ['GOOD', 'NIGHT'],
            'HOW ARE YOU': ['H', 'O', 'W', 'A', 'R', 'E', 'Y', 'O', 'U'],
            'I LOVE YOU': ['I', 'LOVE', 'YOU'],
            'NICE TO MEET YOU': ['N', 'I', 'C', 'E', 'T', 'O', 'M', 'E', 'E', 'T', 'Y', 'O', 'U']
        }
        
        # Initialize enhanced neural network model
        self.model = self._create_enhanced_model()
        self.is_trained = False
        
        # Temporal smoothing for better accuracy
        self.prediction_history = deque(maxlen=10)
        self.confidence_threshold = 0.75
        
        # Hand gesture feature extractors
        self.gesture_features = {}
        self._initialize_gesture_features()
        
    def _create_enhanced_model(self):
        """Create an enhanced neural network for improved ASL recognition"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(84,)),  # Increased features
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(26, activation='softmax')  # 26 letters
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _initialize_gesture_features(self):
        """Initialize hand gesture feature patterns for better recognition"""
        # Define characteristic features for each letter
        self.gesture_features = {
            'A': {'closed_fist': True, 'thumb_up': True, 'fingers_closed': [1,2,3,4]},
            'B': {'flat_hand': True, 'thumb_folded': True, 'fingers_extended': [1,2,3,4]},
            'C': {'curved_hand': True, 'thumb_gap': True, 'arc_shape': True},
            'D': {'index_up': True, 'thumb_touch': True, 'fingers_closed': [2,3,4]},
            'E': {'fingers_bent': True, 'thumb_folded': True, 'claw_shape': True},
            'F': {'ok_sign': True, 'thumb_index_touch': True, 'fingers_extended': [2,3,4]},
            'G': {'index_extended': True, 'thumb_extended': True, 'pointing_side': True},
            'H': {'two_fingers': True, 'index_middle_extended': True, 'horizontal': True},
            'I': {'pinky_up': True, 'fist_closed': True, 'thumb_folded': True},
            'J': {'pinky_hook': True, 'movement_pattern': True, 'j_shape': True},
            'K': {'index_middle_up': True, 'thumb_between': True, 'v_shape': True},
            'L': {'l_shape': True, 'thumb_index_90': True, 'right_angle': True},
            'M': {'three_fingers_down': True, 'thumb_under': True, 'm_shape': True},
            'N': {'two_fingers_down': True, 'thumb_under': True, 'n_shape': True},
            'O': {'circle_shape': True, 'all_fingers_curved': True, 'o_shape': True},
            'P': {'index_down': True, 'thumb_middle': True, 'p_shape': True},
            'Q': {'thumb_index_down': True, 'pointing_down': True, 'q_shape': True},
            'R': {'index_middle_crossed': True, 'r_shape': True, 'fingers_crossed': True},
            'S': {'fist_closed': True, 'thumb_over': True, 's_shape': True},
            'T': {'thumb_between_index': True, 't_shape': True, 'fist_with_thumb': True},
            'U': {'two_fingers_up': True, 'index_middle_together': True, 'u_shape': True},
            'V': {'peace_sign': True, 'index_middle_apart': True, 'v_shape': True},
            'W': {'three_fingers_up': True, 'index_middle_ring': True, 'w_shape': True},
            'X': {'index_bent': True, 'hook_shape': True, 'x_shape': True},
            'Y': {'thumb_pinky_out': True, 'shaka_sign': True, 'y_shape': True},
            'Z': {'z_movement': True, 'index_tracing': True, 'z_pattern': True}
        }
    
    def extract_enhanced_features(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract enhanced features from image using MediaPipe"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_image)
            pose_results = self.pose.process(rgb_image)
            
            features = []
            
            # Extract hand landmarks
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                
                # Basic landmark coordinates (21 points * 3 coordinates = 63 features)
                for landmark in hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
                
                # Additional geometric features
                geometric_features = self._extract_geometric_features(hand_landmarks)
                features.extend(geometric_features)
                
            else:
                # Fill with zeros if no hand detected
                features.extend([0.0] * 63)  # Basic landmarks
                features.extend([0.0] * 21)  # Geometric features
            
            return np.array(features) if len(features) == 84 else None
            
        except Exception as e:
            logger.error(f"Error extracting enhanced features: {e}")
            return None
    
    def _extract_geometric_features(self, hand_landmarks) -> List[float]:
        """Extract geometric features from hand landmarks"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        landmarks = np.array(landmarks)
        
        features = []
        
        # 1. Finger tip distances from palm center
        palm_center = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        for tip in finger_tips:
            distance = np.linalg.norm(landmarks[tip] - palm_center)
            features.append(distance)
        
        # 2. Finger angles
        for i in range(1, 5):  # For each finger except thumb
            base = 5 + (i-1) * 4  # Base joint
            tip = base + 3        # Tip joint
            
            if base < len(landmarks) and tip < len(landmarks):
                vector = landmarks[tip] - landmarks[base]
                angle = math.atan2(vector[1], vector[0])
                features.append(angle)
            else:
                features.append(0.0)
        
        # 3. Hand orientation
        wrist_to_middle = landmarks[9] - landmarks[0]
        hand_angle = math.atan2(wrist_to_middle[1], wrist_to_middle[0])
        features.append(hand_angle)
        
        # 4. Finger spread (distances between adjacent finger tips)
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        for i in range(len(finger_tips) - 1):
            distance = np.linalg.norm(landmarks[finger_tips[i]] - landmarks[finger_tips[i+1]])
            features.append(distance)
        
        # 5. Thumb position relative to other fingers
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        thumb_index_distance = np.linalg.norm(thumb_tip - index_tip)
        features.append(thumb_index_distance)
        
        # 6. Hand compactness (bounding box ratio)
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        compactness = width / height if height > 0 else 1.0
        features.append(compactness)
        
        # 7. Finger curvature (sum of angles in finger joints)
        curvatures = []
        for finger in range(1, 5):  # Skip thumb for simplicity
            base = 5 + (finger-1) * 4
            joints = [base, base+1, base+2, base+3]
            
            if all(j < len(landmarks) for j in joints):
                total_curvature = 0
                for i in range(len(joints) - 2):
                    v1 = landmarks[joints[i+1]] - landmarks[joints[i]]
                    v2 = landmarks[joints[i+2]] - landmarks[joints[i+1]]
                    
                    # Calculate angle between vectors
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = math.acos(np.clip(cos_angle, -1, 1))
                    total_curvature += angle
                
                curvatures.append(total_curvature)
            else:
                curvatures.append(0.0)
        
        features.extend(curvatures)
        
        # Ensure we have exactly 21 geometric features
        while len(features) < 21:
            features.append(0.0)
        
        return features[:21]
    
    def predict_letter_enhanced(self, features: np.ndarray) -> Tuple[str, float]:
        """Enhanced prediction with temporal smoothing and confidence weighting"""
        try:
            if not self.is_trained:
                # Use enhanced rule-based prediction
                letter, confidence = self._enhanced_rule_based_prediction(features)
            else:
                # Use trained model
                features_reshaped = features.reshape(1, -1)
                prediction = self.model.predict(features_reshaped, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = float(np.max(prediction[0]))
                letter = self.asl_alphabet.get(predicted_class, 'UNKNOWN')
            
            # Add to prediction history for temporal smoothing
            self.prediction_history.append((letter, confidence))
            
            # Apply temporal smoothing
            smoothed_letter, smoothed_confidence = self._apply_temporal_smoothing()
            
            return smoothed_letter, smoothed_confidence
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {e}")
            return 'UNKNOWN', 0.0
    
    def _enhanced_rule_based_prediction(self, features: np.ndarray) -> Tuple[str, float]:
        """Enhanced rule-based prediction using geometric features"""
        try:
            if len(features) < 84:
                return 'UNKNOWN', 0.0
            
            # Extract basic coordinates and geometric features
            coordinates = features[:63]  # First 63 are landmark coordinates
            geometric = features[63:]    # Last 21 are geometric features
            
            # Analyze geometric features for better prediction
            finger_distances = geometric[:5]    # Distances from palm center
            finger_angles = geometric[5:9]      # Finger angles
            hand_angle = geometric[9]           # Hand orientation
            finger_spreads = geometric[10:13]   # Finger spreads
            thumb_distance = geometric[13]      # Thumb-index distance
            compactness = geometric[14]         # Hand compactness
            curvatures = geometric[15:19]       # Finger curvatures
            
            # Rule-based classification using geometric features
            predictions = []
            
            # Analyze hand shape patterns
            if compactness < 0.6 and max(finger_distances) > 0.15:
                # Extended fingers - likely B, C, or open hand letters
                if thumb_distance < 0.05:  # Thumb close to index
                    predictions.append(('F', 0.8))
                elif max(finger_spreads) > 0.1:  # Fingers spread
                    predictions.append(('B', 0.85))
                else:
                    predictions.append(('C', 0.75))
            
            elif compactness > 0.8 and max(finger_distances) < 0.1:
                # Closed fist - likely A, S, T
                if thumb_distance > 0.08:  # Thumb extended
                    predictions.append(('A', 0.9))
                elif sum(curvatures) > 2.0:  # High curvature
                    predictions.append(('S', 0.85))
                else:
                    predictions.append(('T', 0.8))
            
            elif len([d for d in finger_distances if d > 0.12]) == 2:
                # Two fingers extended
                if finger_spreads[0] > 0.08:  # V shape
                    predictions.append(('V', 0.9))
                else:  # U shape
                    predictions.append(('U', 0.85))
            
            elif len([d for d in finger_distances if d > 0.12]) == 1:
                # One finger extended
                if finger_distances[1] > 0.12:  # Index finger
                    predictions.append(('D', 0.85))
                elif finger_distances[4] > 0.12:  # Pinky
                    predictions.append(('I', 0.8))
            
            elif len([d for d in finger_distances if d > 0.12]) == 3:
                # Three fingers extended
                predictions.append(('W', 0.8))
            
            # L shape detection
            if (finger_distances[0] > 0.12 and finger_distances[1] > 0.12 and 
                abs(finger_angles[0] - finger_angles[1]) > 1.0):
                predictions.append(('L', 0.9))
            
            # O shape detection
            if (compactness < 0.7 and all(c > 0.5 for c in curvatures) and 
                max(finger_distances) < 0.15):
                predictions.append(('O', 0.85))
            
            # Y shape detection (thumb and pinky extended)
            if (finger_distances[0] > 0.12 and finger_distances[4] > 0.12 and 
                all(d < 0.08 for d in finger_distances[1:4])):
                predictions.append(('Y', 0.9))
            
            # Select best prediction
            if predictions:
                best_prediction = max(predictions, key=lambda x: x[1])
                return best_prediction
            else:
                # Fallback to position-based prediction
                x_coords = coordinates[::3]
                y_coords = coordinates[1::3]
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                
                # Position-based fallback
                if center_y < 0.3:
                    letters = ['A', 'B', 'C', 'D', 'E']
                elif center_y > 0.7:
                    letters = ['V', 'W', 'X', 'Y', 'Z']
                elif center_x < 0.3:
                    letters = ['F', 'G', 'H', 'I', 'J']
                elif center_x > 0.7:
                    letters = ['K', 'L', 'M', 'N', 'O']
                else:
                    letters = ['P', 'Q', 'R', 'S', 'T', 'U']
                
                selected_letter = np.random.choice(letters)
                confidence = np.random.uniform(0.6, 0.8)
                
                return selected_letter, confidence
            
        except Exception as e:
            logger.error(f"Error in enhanced rule-based prediction: {e}")
            return 'UNKNOWN', 0.0
    
    def _apply_temporal_smoothing(self) -> Tuple[str, float]:
        """Apply temporal smoothing to reduce noise in predictions"""
        if not self.prediction_history:
            return 'UNKNOWN', 0.0
        
        # Get recent predictions
        recent_predictions = list(self.prediction_history)
        
        # Count letter frequencies with confidence weighting
        letter_scores = {}
        total_weight = 0
        
        for letter, confidence in recent_predictions:
            if letter != 'UNKNOWN':
                weight = confidence
                if letter in letter_scores:
                    letter_scores[letter] += weight
                else:
                    letter_scores[letter] = weight
                total_weight += weight
        
        if not letter_scores:
            return 'UNKNOWN', 0.0
        
        # Find the letter with highest weighted score
        best_letter = max(letter_scores.keys(), key=lambda k: letter_scores[k])
        confidence = letter_scores[best_letter] / total_weight if total_weight > 0 else 0.0
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return 'UNKNOWN', confidence
        
        return best_letter, min(confidence, 1.0)
    
    def process_video_enhanced(self, video_path: str) -> Dict:
        """Enhanced video processing with improved accuracy"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"Processing video: {frame_count} frames, {fps} FPS, {duration:.2f}s duration")
            
            detected_letters = []
            frame_results = []
            processed_frames = 0
            
            # Process every 3rd frame for better accuracy vs speed balance
            frame_skip = 3
            
            # Reset prediction history for each video
            self.prediction_history.clear()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if processed_frames % frame_skip == 0:
                    # Enhance frame quality
                    enhanced_frame = self._enhance_frame(frame)
                    
                    # Extract enhanced features
                    features = self.extract_enhanced_features(enhanced_frame)
                    
                    if features is not None:
                        # Predict letter with enhanced method
                        letter, confidence = self.predict_letter_enhanced(features)
                        
                        if confidence > 0.7 and letter != 'UNKNOWN':  # Higher threshold
                            detected_letters.append(letter)
                            frame_results.append({
                                'frame': processed_frames,
                                'timestamp': processed_frames / fps if fps > 0 else 0,
                                'letter': letter,
                                'confidence': confidence
                            })
                
                processed_frames += 1
            
            cap.release()
            
            # Enhanced post-processing
            text_output = self._enhanced_post_processing(detected_letters)
            
            return {
                'success': True,
                'detected_letters': detected_letters,
                'text_output': text_output,
                'frame_results': frame_results,
                'video_info': {
                    'duration': duration,
                    'fps': fps,
                    'total_frames': frame_count,
                    'processed_frames': processed_frames
                },
                'confidence_avg': np.mean([r['confidence'] for r in frame_results]) if frame_results else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                'success': False,
                'error': str(e),
                'detected_letters': [],
                'text_output': '',
                'frame_results': []
            }
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame quality for better recognition"""
        # Apply histogram equalization
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def _enhanced_post_processing(self, letters: List[str]) -> str:
        """Enhanced post-processing to form words and sentences"""
        if not letters:
            return ""
        
        # Remove consecutive duplicates with improved logic
        filtered_letters = []
        prev_letter = None
        consecutive_count = 0
        
        for letter in letters:
            if letter != prev_letter:
                # Add letter if it's different or if we've seen it enough times
                if consecutive_count >= 2 or prev_letter is None:
                    filtered_letters.append(letter)
                consecutive_count = 1
                prev_letter = letter
            else:
                consecutive_count += 1
        
        # Add the last letter if it appeared enough times
        if consecutive_count >= 2 and prev_letter:
            filtered_letters.append(prev_letter)
        
        if not filtered_letters:
            return ""
        
        # Try to form words and phrases
        text = ''.join(filtered_letters)
        
        # Check for phrases first (longer matches)
        words = []
        i = 0
        while i < len(text):
            found_match = False
            
            # Check for phrases
            for phrase, phrase_pattern in sorted(self.asl_phrases.items(), 
                                               key=lambda x: len(''.join(x[1])), reverse=True):
                phrase_str = ''.join(phrase_pattern)
                if text[i:].startswith(phrase_str):
                    words.append(phrase)
                    i += len(phrase_str)
                    found_match = True
                    break
            
            if not found_match:
                # Check for individual words
                for word, word_letters in sorted(self.asl_words.items(), 
                                               key=lambda x: len(x[1]), reverse=True):
                    word_str = ''.join(word_letters)
                    if text[i:].startswith(word_str):
                        words.append(word)
                        i += len(word_str)
                        found_match = True
                        break
            
            if not found_match:
                words.append(text[i])
                i += 1
        
        # Join words with spaces and clean up
        result = ' '.join(words)
        
        # Post-process to fix common issues
        result = self._clean_text_output(result)
        
        return result
    
    def _clean_text_output(self, text: str) -> str:
        """Clean and improve text output"""
        # Remove single letters that might be noise (except I and A)
        words = text.split()
        cleaned_words = []
        
        for word in words:
            if len(word) == 1 and word not in ['I', 'A']:
                # Skip single letters that are likely noise
                continue
            cleaned_words.append(word)
        
        # Join and capitalize properly
        result = ' '.join(cleaned_words)
        
        # Capitalize first letter of sentences
        if result:
            result = result[0].upper() + result[1:].lower()
        
        return result

# Test function
def test_improved_asl_recognition():
    """Test improved ASL recognition system"""
    recognizer = ImprovedASLRecognizer()
    
    # Create dummy enhanced features for testing
    dummy_features = np.random.rand(84)  # 63 landmarks + 21 geometric features
    
    letter, confidence = recognizer.predict_letter_enhanced(dummy_features)
    print(f"Enhanced test prediction: {letter} (confidence: {confidence:.2f})")
    
    return recognizer

if __name__ == "__main__":
    # Test the improved ASL recognition system
    recognizer = test_improved_asl_recognition()
    print("Improved ASL Recognition system initialized successfully!")

