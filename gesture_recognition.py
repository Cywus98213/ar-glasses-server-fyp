#!/usr/bin/env python3

import os
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[GESTURE] Warning: MediaPipe not available. Install with: pip install mediapipe")


class GestureRecognizer:
    """MediaPipe-based gesture recognition for sign language."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the gesture recognizer.
        
        Args:
            model_path: Path to custom trained MediaPipe gesture model (.task file)
        """
        print("[GESTURE] Initializing Gesture Recognizer...")
        
        if not MEDIAPIPE_AVAILABLE:
            print("[GESTURE] ERROR: MediaPipe is not installed!")
            print("[GESTURE] Install with: pip install mediapipe")
            self.gesture_recognizer = None
            return
        
        # Default model path if not provided
        if model_path is None:
            model_path = "sign_gesture_model/gesture_recognizer.task"
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f"[GESTURE] ERROR: Model file not found at {model_path}")
            print("[GESTURE] Gesture recognition will be disabled")
            self.gesture_recognizer = None
            return
        
        try:
            # Initialize MediaPipe gesture recognizer
            base_options = python.BaseOptions(model_asset_path=str(model_path))
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,  # Process single images
                num_hands=2,  # Detect up to 2 hands
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
            print(f"[GESTURE] Gesture model loaded from: {model_path}")
            print("[GESTURE] Gesture recognition enabled")
            
        except Exception as e:
            print(f"[GESTURE] ERROR: Could not load gesture model: {e}")
            import traceback
            print(f"[GESTURE] Traceback: {traceback.format_exc()}")
            self.gesture_recognizer = None
            print("[GESTURE] Gesture recognition will be disabled")
    
    def recognize_gesture(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Recognize gesture from image data.
        
        Args:
            image_data: Image as numpy array (BGR format from OpenCV or RGB)
            
        Returns:
            Dictionary containing:
                - 'gestures': List of detected gestures with confidence
                - 'hand_landmarks': Hand landmark data
                - 'success': Boolean indicating if recognition was successful
        """
        if self.gesture_recognizer is None:
            return {
                'success': False,
                'error': 'Gesture recognizer not initialized',
                'gestures': [],
                'hand_landmarks': []
            }
        
        try:
            # Ensure image is in the correct format for MediaPipe
            # MediaPipe expects RGB format (uint8)
            
            # Handle different input formats
            if len(image_data.shape) == 2:
                # Grayscale - convert to RGB
                rgb_image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
            elif len(image_data.shape) == 3:
                if image_data.shape[2] == 4:
                    # RGBA - convert to RGB
                    rgb_image = cv2.cvtColor(image_data, cv2.COLOR_RGBA2RGB)
                elif image_data.shape[2] == 3:
                    # Check if it's already RGB (from PIL) or BGR (from OpenCV)
                    # PIL images are RGB, OpenCV images are BGR
                    # We'll assume RGB if it comes from recognize_gesture_from_base64 (PIL)
                    # For safety, we'll check the first pixel - but actually, let's just assume RGB
                    # since we're converting PIL images to RGB explicitly
                    rgb_image = image_data
                else:
                    rgb_image = image_data
            else:
                raise ValueError(f"Unsupported image shape: {image_data.shape}")
            
            # Ensure dtype is uint8
            if rgb_image.dtype != np.uint8:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image = rgb_image.astype(np.uint8)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Recognize gesture
            recognition_result = self.gesture_recognizer.recognize(mp_image)
            
            gestures = []
            hand_landmarks = []
            
            # Process recognition results
            if recognition_result.gestures:
                for gesture_list in recognition_result.gestures:
                    if gesture_list:
                        for gesture in gesture_list:
                            gestures.append({
                                'category_name': gesture.category_name,
                                'score': float(gesture.score)
                            })
            
            # Extract hand landmarks
            if recognition_result.hand_landmarks:
                for hand_landmark_list in recognition_result.hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmark_list:
                        landmarks.append({
                            'x': float(landmark.x),
                            'y': float(landmark.y),
                            'z': float(landmark.z)
                        })
                    hand_landmarks.append(landmarks)
            
            return {
                'success': True,
                'gestures': gestures,
                'hand_landmarks': hand_landmarks,
                'num_hands': len(hand_landmarks)
            }
            
        except Exception as e:
            print(f"[GESTURE] Error recognizing gesture: {e}")
            import traceback
            print(f"[GESTURE] Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'gestures': [],
                'hand_landmarks': []
            }
    
    def recognize_gesture_from_base64(self, base64_data: str) -> Dict[str, Any]:
        """
        Recognize gesture from base64-encoded image.
        
        Args:
            base64_data: Base64-encoded image string
            
        Returns:
            Dictionary with recognition results
        """
        try:
            import base64
            from io import BytesIO
            from PIL import Image
            
            # Decode base64 image
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Convert PIL Image to RGB if needed (handles RGBA, L, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL Image to numpy array
            image_array = np.array(image)
            
            # Ensure it's uint8
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # Recognize gesture
            return self.recognize_gesture(image_array)
            
        except Exception as e:
            print(f"[GESTURE] Error processing base64 image: {e}")
            import traceback
            print(f"[GESTURE] Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e),
                'gestures': [],
                'hand_landmarks': []
            }
    
    def is_available(self) -> bool:
        """Check if gesture recognition is available."""
        return self.gesture_recognizer is not None
