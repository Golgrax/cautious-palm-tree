import cv2
import numpy as np
import os
from typing import List, Tuple
import math

def create_test_asl_video(output_path: str = "test_asl_video.mp4"):
    """Create a test ASL video for demonstration purposes"""
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 8  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Define hand positions for different letters
    hand_positions = {
        'H': [(200, 200), (250, 200), (200, 250), (250, 250)],  # Two fingers horizontal
        'E': [(220, 220), (220, 240), (220, 260), (220, 280)],  # Fingers bent
        'L': [(180, 200), (180, 300), (250, 200), (180, 200)],  # L shape
        'O': [(200, 220), (240, 200), (260, 240), (240, 280), (200, 260), (160, 240)],  # Circle
        'W': [(180, 200), (200, 180), (220, 200), (240, 180)],  # Three fingers
        'R': [(200, 200), (220, 180), (200, 250), (180, 230)],  # Crossed fingers
        'D': [(200, 200), (200, 280), (220, 200), (180, 220)],  # Index up
        'T': [(200, 200), (200, 250), (180, 225), (220, 225)],  # Thumb between
        'A': [(200, 220), (180, 240), (220, 240), (200, 260)],  # Fist with thumb
        'N': [(200, 220), (180, 240), (200, 260), (220, 240)],  # Two fingers down
        'K': [(200, 200), (220, 180), (200, 220), (180, 200)],  # V with thumb
        'Y': [(180, 200), (220, 200), (200, 240), (200, 180)],  # Thumb and pinky
        'U': [(190, 200), (210, 200), (190, 250), (210, 250)]   # Two fingers together
    }
    
    # Text sequence to demonstrate
    text_sequence = "HELLO WORLD THANK YOU"
    letters = [c for c in text_sequence if c.isalpha()]
    
    frames_per_letter = total_frames // len(letters)
    
    print(f"Creating test ASL video: {len(letters)} letters, {frames_per_letter} frames per letter")
    
    for frame_idx in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Determine current letter
        letter_idx = min(frame_idx // frames_per_letter, len(letters) - 1)
        current_letter = letters[letter_idx]
        
        # Get hand position for current letter
        if current_letter in hand_positions:
            positions = hand_positions[current_letter]
            
            # Add some animation/movement
            animation_offset = int(10 * math.sin(frame_idx * 0.2))
            
            # Draw hand landmarks as circles
            for i, (x, y) in enumerate(positions):
                # Apply animation offset
                animated_x = x + animation_offset
                animated_y = y + int(5 * math.cos(frame_idx * 0.15 + i))
                
                # Draw landmark point
                cv2.circle(frame, (animated_x, animated_y), 8, (0, 255, 0), -1)
                cv2.circle(frame, (animated_x, animated_y), 12, (255, 255, 255), 2)
                
                # Add landmark number
                cv2.putText(frame, str(i), (animated_x - 5, animated_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Connect landmarks with lines to show hand structure
            if len(positions) >= 4:
                for i in range(len(positions) - 1):
                    x1, y1 = positions[i]
                    x2, y2 = positions[i + 1]
                    cv2.line(frame, (x1 + animation_offset, y1), 
                            (x2 + animation_offset, y2), (0, 200, 0), 2)
        
        # Add text overlay showing current letter
        cv2.putText(frame, f"Letter: {current_letter}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Add progress indicator
        progress = (frame_idx + 1) / total_frames
        cv2.rectangle(frame, (50, height - 50), (int(50 + 300 * progress), height - 30), 
                     (0, 255, 255), -1)
        cv2.rectangle(frame, (50, height - 50), (350, height - 30), (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames}", (400, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add expected text
        cv2.putText(frame, f"Expected: {text_sequence}", (50, height - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Test ASL video created: {output_path}")
    return output_path

def create_advanced_test_video(output_path: str = "advanced_asl_test.mp4"):
    """Create a more advanced test video with realistic hand movements"""
    
    width, height = 640, 480
    fps = 30
    duration = 10
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # More realistic hand landmark positions (21 points for MediaPipe)
    def get_hand_landmarks(letter: str, frame_offset: int = 0) -> List[Tuple[int, int]]:
        """Get 21 hand landmark positions for a given letter"""
        base_positions = {
            'A': [  # Closed fist with thumb up
                (200, 250),  # Wrist
                (180, 230), (170, 210), (165, 190), (160, 170),  # Thumb
                (220, 220), (230, 200), (235, 180), (240, 160),  # Index
                (240, 220), (250, 200), (255, 180), (260, 160),  # Middle
                (260, 220), (270, 200), (275, 180), (280, 160),  # Ring
                (280, 230), (285, 210), (290, 190), (295, 170)   # Pinky
            ],
            'B': [  # Flat hand
                (200, 250),  # Wrist
                (170, 240), (160, 230), (155, 220), (150, 210),  # Thumb
                (220, 220), (225, 180), (230, 140), (235, 100),  # Index
                (240, 220), (245, 180), (250, 140), (255, 100),  # Middle
                (260, 220), (265, 180), (270, 140), (275, 100),  # Ring
                (280, 220), (285, 180), (290, 140), (295, 100)   # Pinky
            ],
            'L': [  # L shape
                (200, 250),  # Wrist
                (160, 230), (140, 210), (120, 190), (100, 170),  # Thumb extended
                (220, 220), (225, 180), (230, 140), (235, 100),  # Index up
                (240, 240), (245, 250), (250, 260), (255, 270),  # Middle folded
                (260, 240), (265, 250), (270, 260), (275, 270),  # Ring folded
                (280, 240), (285, 250), (290, 260), (295, 270)   # Pinky folded
            ]
        }
        
        if letter not in base_positions:
            letter = 'A'  # Default fallback
        
        positions = base_positions[letter]
        
        # Add slight animation
        animated_positions = []
        for i, (x, y) in enumerate(positions):
            offset_x = int(3 * math.sin(frame_offset * 0.1 + i * 0.5))
            offset_y = int(2 * math.cos(frame_offset * 0.08 + i * 0.3))
            animated_positions.append((x + offset_x, y + offset_y))
        
        return animated_positions
    
    # Test sequence
    test_letters = ['A', 'B', 'L', 'A', 'B', 'L']
    frames_per_letter = total_frames // len(test_letters)
    
    print(f"Creating advanced test video: {len(test_letters)} letters")
    
    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Current letter
        letter_idx = min(frame_idx // frames_per_letter, len(test_letters) - 1)
        current_letter = test_letters[letter_idx]
        
        # Get hand landmarks
        landmarks = get_hand_landmarks(current_letter, frame_idx)
        
        # Draw hand landmarks
        for i, (x, y) in enumerate(landmarks):
            if i == 0:  # Wrist - larger circle
                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            else:  # Other joints
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
        
        # Draw hand connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_pos = landmarks[start_idx]
                end_pos = landmarks[end_idx]
                cv2.line(frame, start_pos, end_pos, (255, 255, 255), 2)
        
        # Add information overlay
        cv2.putText(frame, f"Letter: {current_letter}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        
        cv2.putText(frame, f"Frame: {frame_idx + 1}/{total_frames}", (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.putText(frame, "Advanced ASL Test Video", (50, height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Advanced test video created: {output_path}")
    return output_path

if __name__ == "__main__":
    # Create test videos
    basic_video = create_test_asl_video("/home/ubuntu/asl-translator/uploads/test_asl_basic.mp4")
    advanced_video = create_advanced_test_video("/home/ubuntu/asl-translator/uploads/test_asl_advanced.mp4")
    
    print("Test videos created successfully!")
    print(f"Basic: {basic_video}")
    print(f"Advanced: {advanced_video}")

