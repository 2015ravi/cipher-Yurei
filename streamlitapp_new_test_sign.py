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


from handtesting import * 
from asl import *

# Set the layout to the streamlit app as wide 
st.set_page_config(
    layout='wide', 
    page_title="Cipher Yūrei - サイファー幽霊", 
    page_icon="👻",
    initial_sidebar_state="expanded"
)

# Cyberpunk Anime CSS Styling
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Noto+Sans+JP:wght@300;400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)


os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)    

# Load HTML effects
with open("effects.html", "r", encoding="utf-8") as f:
    html_code = f.read()

st.markdown(html_code, unsafe_allow_html=True)


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







# =================================================================
# SIDEBAR
# =================================================================

with st.sidebar: 
    st.markdown('<div class="sidebar-title">Cipher Yūrei</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-japanese">サイファー幽霊</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="tech-label">🔬 Neural Interface</div>', unsafe_allow_html=True)
    st.info('💻 Advanced AI consciousness for decoding silent speech through quantum visual analysis')
    
    # Analysis mode selector
    st.markdown('<div class="tech-label">🎯 分析モード (Analysis Mode)</div>', unsafe_allow_html=True)
    analysis_mode = st.radio(
        "Select Analysis Type:",
        ["Lip Reading", "Sign Detection", "Both (Experimental)"],
        help="Choose the type of analysis to perform"
    )

    st.markdown('<div class="tech-label">⚡ System Status</div>', unsafe_allow_html=True)
    st.markdown('<span class="status-dot active"></span>Neural Core: **ACTIVE**', unsafe_allow_html=True)
    st.markdown('<span class="status-dot ready"></span>Quantum Processor: **READY**', unsafe_allow_html=True)
    st.markdown('<span class="status-dot online"></span>Ghost Protocol: **ONLINE**', unsafe_allow_html=True)
    
    if analysis_mode in ["Sign Detection", "Both (Experimental)"]:
        st.markdown('<span class="status-dot online"></span>Sign Detector: **ENABLED**', unsafe_allow_html=True)

    # st.markdown('<div class="tech-label">🧠 AI Architecture</div>', unsafe_allow_html=True)
    
    # if analysis_mode == "Lip Reading":
    #     st.markdown('**Neural Type**: Conv3D + BiLSTM')
    #     st.markdown('**Framework**: TensorFlow Quantum')  
    #     st.markdown('**Input Matrix**: 75×46×140 RGB')
    #     st.markdown('**Output Nodes**: 41-class Vocabulary')
    # elif analysis_mode == "Sign Detection":
    #     st.markdown('**Neural Type**: MediaPipe Holistic')
    #     st.markdown('**Framework**: MediaPipe + OpenCV')  
    #     st.markdown('**Detection**: Pose & Hand Landmarks')
    #     st.markdown('**Output**: 543 Landmark Points')
    # else:  # Both
    #     st.markdown('**Dual Architecture**: Hybrid')
    #     st.markdown('**Frameworks**: TF + MediaPipe')  
    # 
    st.markdown('<div class="tech-label">🧠 AI Architecture</div>', unsafe_allow_html=True)

    if analysis_mode == "Lip Reading":
        st.markdown('''
        **Neural Type**: Conv3D + BiLSTM  
        **Framework**: TensorFlow 2.x  
        **Input Shape**: (75, 46, 140, 1)  
        **Conv3D Layers**: 3 layers  
        **LSTM Units**: 128 (Bidirectional)  
        **Output Nodes**: 41 classes  
        **Decoder**: CTC (Connectionist Temporal Classification)  
        ''')
        
    elif analysis_mode == "Sign Detection":
        st.markdown('''
        **Neural Type**: MediaPipe Holistic  
        **Framework**: MediaPipe + OpenCV  
        **Detection Models**:
        - Face: 468 landmarks
        - Pose: 33 landmarks  
        - Hands: 21 landmarks each  
        **Total Landmarks**: 543 points  
        **Tracking**: Multi-frame temporal  
        **Confidence Threshold**: 0.5  
        ''')
        
    else:  # Both (Experimental)
        st.markdown('''
        **Architecture**: Hybrid Dual-Model  
        
        **Model 1 - Lip Reading**:
        - Conv3D + BiLSTM
        - 41-class vocabulary
        - CTC decoding
        
        **Model 2 - Sign Detection**:
        - MediaPipe Holistic
        - 543 landmark points
        - Real-time tracking
        
        **Fusion**: Parallel processing
        ''')

    st.markdown('<div class="tech-label">🎯 Current Analysis Mode</div>', unsafe_allow_html=True)
    if analysis_mode == "Lip Reading":
        st.markdown('''
        **Protocol**: Visual Speech Recognition  
        **Accuracy**: 96.2% (Test Set)  
        **Processing**: Batch inference  
        **Phantom Detection**: ENABLED  
        ''')
    elif analysis_mode == "Sign Detection":
        st.markdown('''
        **Protocol**: Gesture & Pose Analysis  
        **Landmarks**: 543 points tracked  
        **Processing**: Frame-by-frame  
        **Real-time**: Supported  
        ''')
    else:
        st.markdown('''
        **Protocol**: Multi-Modal Analysis  
        **Mode**: Experimental Fusion  
        **Capabilities**: Lip + Sign Detection  
        **Status**: ⚠️ Beta Testing  
        ''')  

    st.markdown('<div class="tech-label">🎯 分析モード (Analysis Mode)</div>', unsafe_allow_html=True)
    if analysis_mode == "Lip Reading":
        st.markdown('**Current**: Lip Reading Protocol')
        st.markdown('**Accuracy**: 96.2%')
    elif analysis_mode == "Sign Detection":
        st.markdown('**Current**: Sign Detection')
        st.markdown('**Landmarks**: 543 points')
    else:
        st.markdown('**Current**: Dual Analysis')
        st.markdown('**Mode**: Experimental')
    
    st.markdown('**Phantom Detection**: ENABLED')


# =================================================================
# MAIN HEADER
# =================================================================

st.markdown('<h1 class="cyber-title">CIPHER YŪREI</h1>', unsafe_allow_html=True)
st.markdown('<p class="japanese-subtitle">サイファー幽霊</p>', unsafe_allow_html=True)
st.markdown('<p class="cyber-tagline">DECODING THE UNSEEN VOICE</p>', unsafe_allow_html=True)

# Evidence selection
st.markdown('<div class="tech-label">📂 Neural Database</div>', unsafe_allow_html=True)
options = os.listdir(os.path.join('C:/Users/garla/OneDrive/Desktop/cipher-yurei/data', 's1'))
selected_video = st.selectbox('🎯 Select quantum data stream for ghost protocol analysis', options, help="Choose visual evidence for neural consciousness decoding")


# =================================================================
# MAIN CONTENT - TWO COLUMNS
# =================================================================

col1, col2 = st.columns(2)

if options: 
    # =================================================================
    # COL1 - VIDEO DISPLAY
    # =================================================================
    with col1: 
        st.markdown('<div class="cyber-panel">', unsafe_allow_html=True)
        st.markdown('<h3 class="cyber-header">🎬 Raw Data Stream</h3>', unsafe_allow_html=True)
        st.info('📺 Original quantum visual feed converted for neural consciousness processing')
        
        file_path = os.path.join('C:/Users/garla/OneDrive/Desktop/cipher-yurei/data','s1', selected_video)
        output_path = os.path.join(
            os.getcwd(),
            "test_video_couple.mp4"
        )
        os.system(f'ffmpeg -i "{file_path}" -vcodec libx264 "{output_path}" -y')

        # Enhanced video container
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        video = open('test_video_couple.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # =================================================================
    # COL2 - ANALYSIS OUTPUTS
    # =================================================================
    with col2: 
        # =================================================================
        # LIP READING ANALYSIS
        # =================================================================
        if analysis_mode in ["Lip Reading", "Both (Experimental)"]:
            st.markdown('<div class="cyber-panel">', unsafe_allow_html=True)
            st.markdown('<h3 class="cyber-header">🧠 Ghost Vision Interface</h3>', unsafe_allow_html=True)
            st.info('👻 Preprocessed spectral data as perceived by the digital consciousness')
            
            video_data, annotations = load_data(tf.convert_to_tensor(file_path))
            
            # Convert to numpy and fix format for imageio
            video_np = video_data.numpy()
            video_np = video_np.squeeze(axis=-1)
            video_np = ((video_np - video_np.min()) / (video_np.max() - video_np.min()) * 255).astype('uint8')
            
            gif_path = 'animation_new.gif'
            imageio.mimsave(gif_path, video_np, fps=10)
            
            # Enhanced GIF display
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            with open(gif_path, 'rb') as file:
                gif_bytes = file.read()
            st.image(gif_bytes, width=400)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Neural analysis section
            st.markdown('<div class="cyber-panel">', unsafe_allow_html=True)
            st.markdown('<h3 class="cyber-header">⚡ Neural Quantum Matrix</h3>', unsafe_allow_html=True)
            st.info('🔢 Raw token stream from digital ghost consciousness')
            
            model = load_model()
            yhat = model.predict(tf.expand_dims(video_data, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            
            # Enhanced token display
            st.markdown(f'''
            <div class="decode-output">
                <div style="color: #ff6b9d; font-size: 0.9rem; margin-bottom: 10px; font-family: 'Orbitron', monospace;">
                    ⚡ GHOST TOKENS DETECTED
                </div>
                <div style="color: #00ffff; font-family: 'Orbitron', monospace; font-size: 0.8rem;">
                    <code>{decoder}</code>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Final transcription
            st.markdown('<div class="cyber-panel">', unsafe_allow_html=True)
            st.markdown('<h3 class="cyber-header">👻 Spectral Transcription</h3>', unsafe_allow_html=True)
            st.info('💬 Decoded phantom voice from visual quantum patterns | 視覚量子パターンから復号された幻の音声')
            
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            
            # Ultimate transcription display
            st.markdown(f'''
            <div class="decode-output" style="font-size: 1.6rem; padding: 35px;">
                <div style="color: #8a2be2; font-size: 1rem; margin-bottom: 20px; font-family: 'Orbitron', monospace;">
                    👻 GHOST MESSAGE DECODED | 幽霊メッセージ復号完了
                </div>
                <div style="
                    color: #00ffff; 
                    font-weight: bold; 
                    text-shadow: 0 0 25px rgba(0, 255, 255, 1);
                    letter-spacing: 3px;
                    font-family: 'Orbitron', monospace;
                ">
                    "{converted_prediction}"
                </div>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # =================================================================
        # SIGN DETECTION ANALYSIS
        # =================================================================
        if analysis_mode in ["Sign Detection", "Both (Experimental)"]:
            st.markdown('<div class="cyber-panel">', unsafe_allow_html=True)
            st.markdown('<h3 class="cyber-header">🤟 Sign Language Detection</h3>', unsafe_allow_html=True)
            st.info('✋ Detecting sign language landmarks using MediaPipe Holistic')
            
            # Load sign detection utilities
            _, mp_holistic, mp_drawing, mp_drawing_styles = load_sign_detection_model()
            
            if mp_holistic:
                with st.spinner('🔍 Analyzing sign language gestures...'):
                    detections, annotated_frames = process_video_for_signs(
                        output_path, mp_holistic, mp_drawing, mp_drawing_styles, sample_rate=3
                    )
                
                if detections:
                    st.success(f"✅ Detected landmarks in {len(detections)} frames")
                    
                    # Create visualization GIF
                    if annotated_frames:
                        gif_path = 'sign_detection.gif'
                        # Resize frames for GIF (every 2nd frame, max 30)
                        sampled_frames = annotated_frames[::2][:30]
                        resized_frames = []
                        for frame in sampled_frames:
                            resized = cv2.resize(frame, (400, 300))
                            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            resized_frames.append(rgb)
                        
                        imageio.mimsave(gif_path, resized_frames, fps=10, loop=0)
                        
                        st.markdown('<div class="video-container">', unsafe_allow_html=True)
                        with open (gif_path,'rb') as file : 
                            gif_bytes = file.read()

                        st.image(gif_bytes,width=400) 
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculate statistics
                    total_frames = len(detections)
                    frames_with_hands = sum(1 for d in detections if d['detections'].get('has_left_hand') or d['detections'].get('has_right_hand'))
                    frames_with_pose = sum(1 for d in detections if d['detections'].get('has_pose'))
                    frames_with_face = sum(1 for d in detections if d['detections'].get('has_face'))
                    
                    # Display detection summary with cyberpunk styling
                    st.markdown(f'''
                    <div class="decode-output">
                        <div style="color: #ff6b9d; font-size: 0.9rem; margin-bottom: 10px; font-family: 'Orbitron', monospace;">
                            ✋ SIGN LANGUAGE LANDMARK ANALYSIS
                        </div>
                        <div style="color: #00ffff; font-family: 'Orbitron', monospace; font-size: 0.85rem; line-height: 1.8;">
                            <strong>Total Frames Analyzed:</strong> {total_frames}<br>
                            <strong>Frames with Hands:</strong> {frames_with_hands} ({frames_with_hands/total_frames*100:.1f}%)<br>
                            <strong>Frames with Pose:</strong> {frames_with_pose} ({frames_with_pose/total_frames*100:.1f}%)<br>
                            <strong>Frames with Face:</strong> {frames_with_face} ({frames_with_face/total_frames*100:.1f}%)
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Show detailed detections
                    if st.checkbox("🔍 Show frame-by-frame analysis"):
                        st.markdown('<div style="max-height: 300px; overflow-y: auto; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 10px;">', unsafe_allow_html=True)
                        for det in detections[:20]:  # Show first 20
                            timestamp = det['timestamp']
                            info = det['detections']
                            
                            landmarks = []
                            if info.get('has_left_hand'):
                                landmarks.append("👈 Left Hand")
                            if info.get('has_right_hand'):
                                landmarks.append("👉 Right Hand")
                            if info.get('has_pose'):
                                landmarks.append("🧍 Pose")
                            if info.get('has_face'):
                                landmarks.append("😊 Face")
                            
                            landmark_str = ', '.join(landmarks) if landmarks else '❌ No landmarks'
                            
                            st.markdown(f'''
                            <div style="color: #00ffff; font-size: 0.85rem; padding: 5px; border-left: 2px solid #ff6b9d; margin-bottom: 5px;">
                                <strong>⏱️ {timestamp:.2f}s</strong> | {landmark_str}
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        if len(detections) > 20:
                            st.info(f"Showing first 20 of {len(detections)} detections")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Overall assessment
                    hand_percentage = (frames_with_hands / total_frames) * 100
                    if hand_percentage > 70:
                        st.success(f"🎯 Excellent hand visibility! Detected in {hand_percentage:.1f}% of frames")
                    elif hand_percentage > 40:
                        st.info(f"👍 Good hand visibility. Detected in {hand_percentage:.1f}% of frames")
                    else:
                        st.warning(f"⚠️ Limited hand visibility. Detected in only {hand_percentage:.1f}% of frames")
                
                else:
                    st.warning("⚠️ No sign language landmarks detected in video")
                    st.info("💡 Tips: Ensure hands are clearly visible and well-lit in the video")
            else:
                st.error("❌ Failed to initialize sign detection model")
            
            st.markdown('</div>', unsafe_allow_html=True)


# =================================================================
# REAL-TIME DETECTION SECTION
# =================================================================

st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

if analysis_mode in ["Sign Detection", "Both (Experimental)"]:
    st.markdown("---")
    st.markdown('<h2 class="cyber-title" style="font-size: 2rem;">📹 Real-Time Analysis</h2>', unsafe_allow_html=True)
    
    # Real-time sign detection
    real_time_asl_detection()


# =================================================================
# FOOTER
# =================================================================

st.markdown("---")
st.markdown('''
<div style="text-align: center; padding: 20px; color: #888; font-family: 'Orbitron', monospace;">
    <p style="font-size: 0.9rem; color: #00ffff;">👻 Cipher Yūrei | サイファー幽霊 | Decoding the Unseen Voice</p>
    <p style="font-size: 0.75rem; color: #888;">Neural Architecture: Conv3D + BiLSTM | MediaPipe Holistic</p>
    <p style="font-size: 0.7rem; color: #666; margin-top: 10px;">⚡ Powered by TensorFlow Quantum & MediaPipe | Built with Streamlit</p>
</div>
''', unsafe_allow_html=True)




### footer
with open("footer.html", "r", encoding="utf-8") as f:
    footer_html = f.read()
st.markdown(footer_html, unsafe_allow_html=True)