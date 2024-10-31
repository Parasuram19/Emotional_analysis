import av
import cv2
import dlib
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from deepface import DeepFace
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import queue
import threading
from typing import List, NamedTuple
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize session state
if 'session_data' not in st.session_state:
    data_columns = [
        "Timestamp", "Face Depression Score", "Face Depression Stage",
        "Euler_Yaw", "Euler_Pitch", "Euler_Roll", "Dominant_Emotion"
    ]
    st.session_state.session_data = pd.DataFrame(columns=data_columns)

# Load face detection and landmark models
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Camera matrix
def get_camera_matrix(frame_size):
    focal_length = frame_size[1]
    center = (frame_size[1] // 2, frame_size[0] // 2)
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

class Frame(NamedTuple):
    frame: np.ndarray
    depression_score: float
    depression_stage: str
    euler_angles: tuple
    dominant_emotion: str

# Queue for frame processing results
result_queue: queue.Queue = queue.Queue()

def get_head_pose(shape, frame_size):
    camera_matrix = get_camera_matrix(frame_size)
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),   # Nose tip
        (shape.part(8).x, shape.part(8).y),     # Chin
        (shape.part(36).x, shape.part(36).y),   # Left eye left corner
        (shape.part(45).x, shape.part(45).y),   # Right eye right corner
        (shape.part(48).x, shape.part(48).y),   # Left mouth corner
        (shape.part(54).x, shape.part(54).y)    # Right mouth corner
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))
    _, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    return rotation_matrix

def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + 
                 rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

def analyze_emotion(frame):
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return results[0] if isinstance(results, list) else results
    except Exception as e:
        logging.error(f"Emotion analysis error: {e}")
        return None

def calculate_depression_score(emotions):
    depression_weights = {
        'sad': 1.0,
        'angry': 0.8,
        'neutral': 0.4,
        'fear': 0.7,
        'happy': -0.5
    }
    score = sum(depression_weights.get(emotion, 0) * emotions[emotion] 
                for emotion in emotions)
    return max(0, min(score * 100, 100))

def get_depression_stage(score):
    if score < 30:
        return "Stage 1: Low"
    elif 30 <= score < 70:
        return "Stage 2: Moderate"
    else:
        return "Stage 3: High"

class VideoProcessor:
    def __init__(self):
        self.frame_count = 0
        self.process_this_frame = True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process every 3rd frame to reduce computational load
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            rotation_matrix = get_head_pose(shape, img.shape[:2])
            euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Get face ROI and analyze emotion
            face_roi = img[face.top():face.bottom(), face.left():face.right()]
            emotion_result = analyze_emotion(face_roi)
            
            if emotion_result is not None:
                depression_score = calculate_depression_score(emotion_result['emotion'])
                depression_stage = get_depression_stage(depression_score)
                dominant_emotion = max(emotion_result['emotion'].items(), 
                                    key=lambda x: x[1])[0]

                # Draw rectangle around face
                cv2.rectangle(img, (face.left(), face.top()), 
                            (face.right(), face.bottom()), (255, 0, 0), 2)

                # Add text annotations
                cv2.putText(img, f"Depression: {depression_stage}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, f"Emotion: {dominant_emotion}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Put result in queue
                result_queue.put(Frame(
                    frame=img,
                    depression_score=depression_score,
                    depression_stage=depression_stage,
                    euler_angles=euler_angles,
                    dominant_emotion=dominant_emotion
                ))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def display_summary_report():
    st.header("Summary Report and Analysis")
    
    if len(st.session_state.session_data) == 0:
        st.warning("No data available for analysis yet.")
        return

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(st.session_state.session_data)

    # Calculate summary statistics
    summary = {
        'Average Depression Score': st.session_state.session_data['Face Depression Score'].mean(),
        'Most Common Depression Stage': st.session_state.session_data['Face Depression Stage'].mode()[0],
        'Most Common Emotion': st.session_state.session_data['Dominant_Emotion'].mode()[0]
    }

    # Display summary statistics
    st.subheader("Summary Statistics")
    for key, value in summary.items():
        st.write(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    # Create visualizations
    st.subheader("Depression Score Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=st.session_state.session_data, 
                x="Timestamp", 
                y="Face Depression Score", 
                marker="o")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Euler angles visualization
    st.subheader("Head Pose Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    euler_data = st.session_state.session_data[['Timestamp', 'Euler_Yaw', 
                                              'Euler_Pitch', 'Euler_Roll']]
    euler_data.set_index('Timestamp').plot(ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def main():
    st.title("Real-Time Depression Analysis")
    st.sidebar.title("Controls")

    # WebRTC Configuration
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Start WebRTC streamer
    ctx = webrtc_streamer(
        key="depression-analysis",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    # Create placeholder for real-time metrics
    metrics_placeholder = st.empty()

    if ctx.state.playing:
        while True:
            try:
                # Get the latest frame result from queue
                frame_result = result_queue.get(timeout=1.0)
                
                # Update session data
                new_row = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Face Depression Score": frame_result.depression_score,
                    "Face Depression Stage": frame_result.depression_stage,
                    "Euler_Yaw": frame_result.euler_angles[0],
                    "Euler_Pitch": frame_result.euler_angles[1],
                    "Euler_Roll": frame_result.euler_angles[2],
                    "Dominant_Emotion": frame_result.dominant_emotion
                }
                
                st.session_state.session_data = pd.concat([
                    st.session_state.session_data,
                    pd.DataFrame([new_row])
                ], ignore_index=True)

                # Update metrics display
                with metrics_placeholder.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Depression Score", 
                                f"{frame_result.depression_score:.1f}%")
                        st.metric("Depression Stage", 
                                frame_result.depression_stage)
                    with col2:
                        st.metric("Dominant Emotion", 
                                frame_result.dominant_emotion)
                        st.metric("Head Pose (Yaw, Pitch, Roll)", 
                                f"{frame_result.euler_angles[0]:.1f}°, "
                                f"{frame_result.euler_angles[1]:.1f}°, "
                                f"{frame_result.euler_angles[2]:.1f}°")

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                break

    # Add button to generate summary report
    if st.button("Generate Summary Report"):
        display_summary_report()

    # Add button to save session data
    if st.button("Save Session Data"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"session_data_{timestamp}.csv"
        st.session_state.session_data.to_csv(filename, index=False)
        st.success(f"Data saved as {filename}")

if __name__ == "__main__":
    main()
