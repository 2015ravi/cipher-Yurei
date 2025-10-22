import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import os
import hashlib
import tempfile

# ============================================================================
# MEDIAPIPE HOLISTIC FEATURE EXTRACTOR
# ============================================================================

class MediaPipeHolisticExtractor:
    """Extract pose, face, and hand landmarks using MediaPipe Holistic"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )
    
    def extract_keypoints(self, image):
        """Extract all keypoints - Returns exactly 1662 features and detection status"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Check if hands are detected
        has_left_hand = results.left_hand_landmarks is not None
        has_right_hand = results.right_hand_landmarks is not None
        has_any_hand = has_left_hand or has_right_hand
        
        # Extract each component
        pose = self._extract_pose(results)           # 132 values
        face = self._extract_face(results)           # 1404 values
        left_hand = self._extract_hand(results.left_hand_landmarks)   # 63 values
        right_hand = self._extract_hand(results.right_hand_landmarks) # 63 values
        
        # Total: 132 + 1404 + 63 + 63 = 1662
        keypoints = np.concatenate([pose, face, left_hand, right_hand])
        
        return keypoints, results, has_any_hand
    
    def _extract_pose(self, results):
        """33 landmarks × 4 (x,y,z,visibility) = 132"""
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] 
                            for lm in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33 * 4)
        return pose
    
    def _extract_face(self, results):
        """468 landmarks × 3 (x,y,z) = 1404"""
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] 
                            for lm in results.face_landmarks.landmark]).flatten()
        else:
            face = np.zeros(468 * 3)
        return face
    
    def _extract_hand(self, hand_landmarks):
        """21 landmarks × 3 (x,y,z) = 63"""
        if hand_landmarks:
            hand = np.array([[lm.x, lm.y, lm.z] 
                            for lm in hand_landmarks.landmark]).flatten()
        else:
            hand = np.zeros(21 * 3)
        return hand
    
    def draw_styled_landmarks(self, image, results):
        """Draw landmarks on image"""
        annotated = image.copy()
        
        # Face mesh
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Left hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Right hand
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return annotated
    
    def close(self):
        self.holistic.close()


# ============================================================================
# LSTM CLASSIFIER
# ============================================================================

class LSTMActionClassifier:
    """LSTM classifier for 3-class ASL gestures"""
    
    def __init__(self, model_path, classes):
        self.classes = classes
        self.model = None
        self.model_loaded = False
        self.sequence_length = 30
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model"""
        try:
            self.model = keras.models.load_model(model_path, compile=False)
            
            # Verify shape
            expected_shape = (None, 30, 1662)
            actual_shape = self.model.input_shape
            
            if actual_shape != expected_shape:
                st.error(f"Model shape mismatch! Expected {expected_shape}, got {actual_shape}")
                return False
            
            # Verify output
            num_classes = self.model.output_shape[-1]
            if num_classes != len(self.classes):
                st.warning(f"Model has {num_classes} outputs but {len(self.classes)} labels provided!")
            
            self.model_loaded = True
            st.success(f"Model loaded successfully!")
            st.info(f"Input: {actual_shape} → Output: {self.model.output_shape}")
            return True
            
        except Exception as e:
            st.error(f"Model load failed: {e}")
            return False
    
    def predict(self, sequence_frames, threshold=0.3):
        """Predict gesture from 30 frames"""
        if not self.model_loaded:
            return None, 0.0, None, "Model not loaded"
        
        if len(sequence_frames) < self.sequence_length:
            needed = self.sequence_length - len(sequence_frames)
            return None, 0.0, None, f"Need {needed} more frames"
        
        try:
            # Take last 30 frames
            sequence = sequence_frames[-self.sequence_length:]
            
            # Convert to numpy array
            sequence_array = np.array(sequence)
            
            # Validate shape
            if sequence_array.shape != (30, 1662):
                return None, 0.0, None, f"Invalid shape: {sequence_array.shape}"
            
            # Check for valid data
            if np.count_nonzero(sequence_array) == 0:
                return None, 0.0, None, "All keypoints are zero - no detection!"
            
            # Add batch dimension: (1, 30, 1662)
            batch = np.expand_dims(sequence_array, axis=0)
            
            # Predict
            predictions = self.model.predict(batch, verbose=0)[0]
            
            # Get result
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            predicted_class = self.classes[class_idx]
            
            # Apply threshold
            if confidence < threshold:
                return None, confidence, predictions, f"Low confidence: {confidence:.2%}"
            
            return predicted_class, confidence, predictions, "Success"
            
        except Exception as e:
            import traceback
            error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
            return None, 0.0, None, error_msg


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize persistent session state"""
    if 'sequence_frames' not in st.session_state:
        st.session_state.sequence_frames = []
    
    if 'frame_images' not in st.session_state:
        st.session_state.frame_images = []
    
    if 'hand_detection_stats' not in st.session_state:
        st.session_state.hand_detection_stats = {}
    
    if 'extractor' not in st.session_state:
        st.session_state.extractor = MediaPipeHolisticExtractor(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )


# ============================================================================
# VIDEO PROCESSING FUNCTION
# ============================================================================

def process_video(video_file, extractor, draw_landmarks=True, min_hand_frames=20):
    """Extract 30 frames from uploaded video with hand detection validation"""
    
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name
    tfile.close()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Failed to open video file!")
        return None, None, None
    
    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st.info(f"Video info: {total_frames} frames at {fps:.1f} FPS ({total_frames/fps:.1f} seconds)")
    
    if total_frames < 30:
        st.error(f"Video too short! Only {total_frames} frames. Need at least 30.")
        cap.release()
        os.unlink(video_path)
        return None, None, None
    
    # Extract all frames
    all_keypoints = []
    all_images = []
    hand_detected_frames = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    frame_idx = 0
    frames_with_hands = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract keypoints with hand detection status
        keypoints, results, has_hand = extractor.extract_keypoints(frame)
        all_keypoints.append(keypoints)
        hand_detected_frames.append(has_hand)
        
        if has_hand:
            frames_with_hands += 1
        
        # Store RGB image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_images.append(rgb_frame)
        
        # Update progress
        frame_idx += 1
        progress = frame_idx / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_idx}/{total_frames} (Hands detected: {frames_with_hands})...")
    
    cap.release()
    os.unlink(video_path)
    
    progress_bar.empty()
    status_text.empty()
    
    # Check if enough frames have hand detection
    hand_detection_rate = frames_with_hands / len(all_keypoints) if all_keypoints else 0
    
    st.info(f"Hand detection: {frames_with_hands}/{len(all_keypoints)} frames ({hand_detection_rate*100:.1f}%)")
    
    if frames_with_hands == 0:
        st.error("❌ **NO HANDS WERE DETECTED in the entire video!**")
        st.warning("Please ensure:")
        st.markdown("- Your hands are clearly visible in the frame")
        st.markdown("- There is adequate lighting")
        st.markdown("- Hands are not too far from the camera")
        st.markdown("- Try moving your hands slower and more deliberately")
        return None, None, {'total': len(all_keypoints), 'with_hands': 0, 'rate': 0}
    
    if frames_with_hands < min_hand_frames:
        st.warning(f"⚠️ Only {frames_with_hands} frames have hand detection (recommended: {min_hand_frames}+)")
        st.info("Predictions may be less accurate. Consider recording a new video with hands more visible.")
    
    if len(all_keypoints) < 30:
        st.error(f"Only extracted {len(all_keypoints)} frames. Need 30.")
        return None, None, {'total': len(all_keypoints), 'with_hands': frames_with_hands, 'rate': hand_detection_rate}
    
    # Sample 30 frames evenly from the video
    indices = np.linspace(0, len(all_keypoints)-1, 30, dtype=int)
    sampled_keypoints = [all_keypoints[i] for i in indices]
    sampled_images = [all_images[i] for i in indices]
    sampled_hand_status = [hand_detected_frames[i] for i in indices]
    
    # Check sampled frames
    sampled_with_hands = sum(sampled_hand_status)
    
    st.success(f"Successfully extracted 30 frames from {len(all_keypoints)} total frames!")
    st.info(f"Of the 30 sampled frames, {sampled_with_hands} have hand detection ({sampled_with_hands/30*100:.1f}%)")
    
    if sampled_with_hands < 15:
        st.warning("⚠️ Less than half of sampled frames have hands detected. Prediction accuracy may be reduced.")
    
    stats = {
        'total': len(all_keypoints),
        'with_hands': frames_with_hands,
        'rate': hand_detection_rate,
        'sampled_total': 30,
        'sampled_with_hands': sampled_with_hands,
        'sampled_rate': sampled_with_hands / 30
    }
    
    return sampled_keypoints, sampled_images, stats


# ============================================================================
# MAIN APP
# ============================================================================

def real_time_asl_detection():
    
    #st.set_page_config(page_title="ASL Detection_VIDEO_UPLOADED", page_icon="🤟", layout="wide")
    initialize_session_state()
    
    st.title("ASL Gesture Detection System")
    st.markdown("**3-Class LSTM Model | 30 Frames | 1662 Features**")
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.header("Configuration")
        
        # Model path
        model_path = 'C:/Users/garla/Downloads/action.h5'
        
        uploaded_model = st.file_uploader(
            "Upload .h5 model file (optional)", 
            type=['h5', 'keras'],
            help="Your trained LSTM model (30×1662 → 3 classes)"
        )
        
        if uploaded_model:
            model_path = 'uploaded_model.h5'
            with open(model_path, 'wb') as f:
                f.write(uploaded_model.getbuffer())
        
        # Action labels
        st.subheader("Action Labels")
        st.warning("MUST match your training order!")
        
        label_input = st.text_area(
            "Enter labels (one per line)",
            value="hello\nthanks\niloveyou",
            height=100,
            help="These are the 3 gestures your model was trained on"
        )
        
        action_labels = [l.strip() for l in label_input.split('\n') if l.strip()]
        
        if len(action_labels) != 3:
            st.error(f"Need exactly 3 labels! You have {len(action_labels)}")
        
        # Settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            0.0, 1.0, 0.3, 0.05,
            help="Lower = more lenient, Higher = stricter"
        )
        
        min_hand_frames = st.slider(
            "Min Frames with Hands",
            10, 30, 20, 1,
            help="Minimum frames that should have hand detection"
        )
        
        draw_landmarks = st.checkbox("Draw Landmarks", value=True)
        
        st.markdown("---")
        
        # Controls
        st.subheader("Controls")
        if st.button("Clear All Frames", use_container_width=True):
            st.session_state.sequence_frames = []
            st.session_state.frame_images = []
            st.session_state.hand_detection_stats = {}
            st.rerun()
        
        st.markdown("---")
        st.caption("Tip: Record 1-2 second video with hands clearly visible")
    
    # Initialize classifier
    if len(action_labels) != 3:
        st.error("Please enter exactly 3 action labels in the sidebar!")
        st.stop()
    
    if 'classifier' not in st.session_state or uploaded_model:
        if model_path and os.path.exists(model_path):
            st.session_state.classifier = LSTMActionClassifier(
                model_path=model_path,
                classes=action_labels
            )
        else:
            st.error("No model file found! Upload a .h5 file in the sidebar.")
            st.info("Your model should accept input shape: (None, 30, 1662)")
            st.stop()
    
    classifier = st.session_state.classifier
    classifier.classes = action_labels
    extractor = st.session_state.extractor
    
    if not classifier.model_loaded:
        st.error("Model failed to load! Check the error messages above.")
        st.stop()
    
    # ========== MAIN INTERFACE ==========
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Upload")
        
        # Progress
        num_frames = len(st.session_state.sequence_frames)
        progress = num_frames / 30
        
        st.progress(min(progress, 1.0))
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Frames", f"{num_frames}/30")
        m2.metric("Progress", f"{min(progress*100, 100):.0f}%")
        
        if num_frames >= 30:
            m3.success("Ready")
        else:
            m3.warning(f"{30-num_frames} needed")
        
        # Show hand detection stats if available
        if st.session_state.hand_detection_stats:
            stats = st.session_state.hand_detection_stats
            st.markdown("---")
            st.markdown("### Hand Detection Statistics")
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Frames", stats.get('total', 0))
            col_b.metric("Frames w/ Hands", stats.get('with_hands', 0))
            col_c.metric("Detection Rate", f"{stats.get('rate', 0)*100:.1f}%")
            
            if 'sampled_with_hands' in stats:
                st.info(f"Sampled frames with hands: {stats['sampled_with_hands']}/30 ({stats['sampled_rate']*100:.1f}%)")
        
        st.markdown("---")
        
        # Video uploader
        st.markdown("### Upload Video of Your Gesture")
        st.info("Record a 1-2 second video performing your ASL gesture")
        
        uploaded_video = st.file_uploader(
            "Choose video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a short video (1-2 seconds) of your gesture"
        )
        
        if uploaded_video is not None:
            st.video(uploaded_video)
            
            if st.button("Process Video", type="primary", use_container_width=True):
                with st.spinner("Extracting frames from video..."):
                    keypoints, images, stats = process_video(
                        uploaded_video, 
                        extractor, 
                        draw_landmarks,
                        min_hand_frames
                    )
                    
                    if keypoints is not None and images is not None:
                        st.session_state.sequence_frames = keypoints
                        st.session_state.frame_images = images
                        st.session_state.hand_detection_stats = stats
                        st.success("Video processed successfully!")
                        st.rerun()
                    else:
                        st.session_state.hand_detection_stats = stats if stats else {}
        
        # Show thumbnails
        if st.session_state.frame_images:
            st.markdown("---")
            st.subheader(f"Extracted Frames ({len(st.session_state.frame_images)})")
            
            cols = st.columns(6)
            for idx, img in enumerate(st.session_state.frame_images[:12]):
                with cols[idx % 6]:
                    st.image(img, caption=f"#{idx+1}", use_column_width=True)
            
            if len(st.session_state.frame_images) > 12:
                st.caption(f"...and {len(st.session_state.frame_images)-12} more frames")
    
    # ========== PREDICTION PANEL ==========
    with col2:
        st.subheader("Prediction")
        
        num_frames = len(st.session_state.sequence_frames)
        
        if num_frames >= 30:
            st.success(f"{num_frames} frames ready!")
            
            # Predict
            if st.button("PREDICT GESTURE", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    action, confidence, all_preds, status = classifier.predict(
                        st.session_state.sequence_frames,
                        threshold=confidence_threshold
                    )
                    
                    if action:
                        # SUCCESS!
                        st.balloons()
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 40px; border-radius: 20px; text-align: center; 
                                    margin: 20px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
                            <div style="color: white; font-size: 3.5rem; font-weight: bold; 
                                        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);">
                                {action.upper()}
                            </div>
                            <div style="color: #e0e0e0; font-size: 1.5rem; margin-top: 15px;">
                                Confidence: {confidence*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success(f"Detected: **{action}** with {confidence*100:.1f}% confidence!")
                    else:
                        st.warning(f"{status}")
                        st.caption("Try: Lower threshold or upload new video")
                    
                    # Show all probabilities
                    if all_preds is not None:
                        st.markdown("---")
                        st.markdown("### All Predictions:")
                        for i, cls in enumerate(classifier.classes):
                            if i < len(all_preds):
                                prob = float(all_preds[i])
                                st.progress(prob)
                                
                                if prob >= confidence_threshold:
                                    st.markdown(f"**{cls}**: {prob*100:.1f}%")
                                else:
                                    st.caption(f"{cls}: {prob*100:.1f}%")
        else:
            st.info(f"Need {30 - num_frames} more frames")
            st.markdown("### Instructions:")
            st.markdown("1. Record 1-2 second video")
            st.markdown("2. Upload video file")
            st.markdown("3. Click 'Process Video'")
            st.markdown("4. Click 'PREDICT'")
        
        # Debug
        st.markdown("---")
        with st.expander("Debug Info"):
            debug_info = {
                'frames_collected': num_frames,
                'frames_needed': max(0, 30 - num_frames),
                'model_loaded': classifier.model_loaded,
                'threshold': confidence_threshold,
                'classes': classifier.classes,
                'hand_detection_stats': st.session_state.hand_detection_stats
            }
            
            st.json(debug_info)
            
            if st.session_state.sequence_frames:
                last_kp = st.session_state.sequence_frames[-1]
                st.write(f"**Last keypoint:**")
                st.write(f"- Shape: {last_kp.shape}")
                st.write(f"- Non-zero: {np.count_nonzero(last_kp)} / {last_kp.shape[0]}")
                st.write(f"- Range: [{last_kp.min():.4f}, {last_kp.max():.4f}]")
    
    # ========== HELP SECTION ==========
    with st.expander("Complete Guide"):
        st.markdown("""
        ## Quick Start Guide
        
        ### 1. Record Video
        - Use your phone or webcam
        - Record 1-2 seconds of your gesture
        - **Keep hands and face visible at all times**
        - Good lighting is important
        - Hands should be clearly visible and not too far away
        
        ### 2. Upload & Process
        - Upload the video file
        - Click "Process Video"
        - App will extract 30 frames automatically
        - **Check hand detection statistics**
        
        ### 3. Predict
        - Click "PREDICT GESTURE"
        - Check confidence score
        - Review all class probabilities
        
        ## Troubleshooting
        
        ### "No Hands Were Detected" Error?
        - ✓ Ensure hands are clearly visible in the frame
        - ✓ Improve lighting (bright, even illumination)
        - ✓ Move hands closer to camera
        - ✓ Perform gesture more slowly and deliberately
        - ✓ Avoid obstructions between hands and camera
        
        ### Low Confidence?
        - Check hand detection rate (should be >70%)
        - Verify lighting is adequate
        - Keep hands clearly visible throughout video
        - Perform gesture slowly and distinctly
        - Lower confidence threshold if needed
        
        ### Wrong Predictions?
        - Verify label order matches training
        - Record longer/clearer video
        - Check detection quality in frames
        - Ensure adequate hand detection in sampled frames
        
        ### Video Won't Process?
        - Try different video format (MP4 works best)
        - Ensure video is at least 1 second long
        - Check video file isn't corrupted
        """)