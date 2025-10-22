# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import cv2
import numpy as np

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model


# Imports for sign detection
import mediapipe as mp
# Add these imports at the top of your file
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch
import torchvision.transforms as transforms
from PIL import Image

import math
import torch.nn as nn
from torchvision.models import resnet50

# =================================================================
# SIGN DETECTION FUNCTIONS
# =================================================================

def load_sign_detection_model():
    """Load MediaPipe Holistic utilities for sign language detection"""
    try:
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create a fresh instance for testing
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        return holistic, mp_holistic, mp_drawing, mp_drawing_styles
    except Exception as e:
        st.error(f"Error loading sign detection model: {e}")
        return None, None, None, None


def detect_hands_in_frame(frame, mp_holistic, mp_drawing, mp_drawing_styles):
    """Detect sign language landmarks in a single frame with fresh holistic instance"""
    holistic = None
    try:
        # Create a NEW holistic instance for this frame
        holistic = mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Process with MediaPipe
        results = holistic.process(frame_rgb)
        
        # Make frame writable
        frame_rgb.flags.writeable = True
        annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        detections = {
            'has_face': results.face_landmarks is not None,
            'has_pose': results.pose_landmarks is not None,
            'has_left_hand': results.left_hand_landmarks is not None,
            'has_right_hand': results.right_hand_landmarks is not None,
        }
        
        return detections, annotated_frame
        
    except Exception as e:
        st.error(f"Error in hand detection: {e}")
        return {}, frame
    finally:
        # CRITICAL: Close the holistic instance
        if holistic is not None:
            holistic.close()


def process_video_for_signs(video_path, mp_holistic, mp_drawing, mp_drawing_styles, sample_rate=5):
    """Process video for sign language detection"""
    cap = cv2.VideoCapture(video_path)
    all_detections = []
    annotated_frames = []
    frame_count = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create ONE holistic instance for the entire video
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame to improve performance
            if frame_count % sample_rate == 0:
                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False
                    
                    # Process with MediaPipe
                    results = holistic.process(frame_rgb)
                    
                    # Make frame writable
                    frame_rgb.flags.writeable = True
                    annotated_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Draw landmarks
                    if results.face_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            results.face_landmarks,
                            mp_holistic.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                    
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            results.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                    
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            results.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            results.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    detections = {
                        'has_face': results.face_landmarks is not None,
                        'has_pose': results.pose_landmarks is not None,
                        'has_left_hand': results.left_hand_landmarks is not None,
                        'has_right_hand': results.right_hand_landmarks is not None,
                    }
                    
                    if any(detections.values()):
                        all_detections.append({
                            'frame': frame_count,
                            'timestamp': frame_count / fps,
                            'detections': detections
                        })
                        annotated_frames.append(annotated_frame)
                    
                except Exception as e:
                    st.warning(f"Error processing frame {frame_count}: {e}")
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            frame_count += 1
    
    finally:
        cap.release()
        holistic.close()
        progress_bar.empty()
        status_text.empty()
    
    return all_detections, annotated_frames