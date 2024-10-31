import cv2
import dlib
import numpy as np
import pyaudio
import wave
import opensmile
import soundfile as sf
import pandas as pd
import joblib
import time
import threading
import streamlit as st
from deepface import DeepFace
import os

# Load OpenCV's pre-trained Haarcascade classifier and dlib predictor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),         # Chin
    (-225.0, 170.0, -135.0),      # Left eye left corner
    (225.0, 170.0, -135.0),       # Right eye right corner
    (-150.0, -150.0, -125.0),     # Left mouth corner
    (150.0, -150.0, -125.0)       # Right mouth corner
])

# Camera matrix
size = (640, 480)
focal_length = size[1]
center = (size[1] // 2, size[0] // 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype="double")

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
FRAME_DURATION = 5  # Duration to process each frame (in seconds)
MODEL_FILENAME = 'random_forest_emodb.pkl'  # Trained model file

# Initialize PyAudio and OpenSMILE
p = pyaudio.PyAudio()
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

# Load the pre-trained model
clf = joblib.load(MODEL_FILENAME)

# Initialize audio-based depression score and emotion label
audio_depression_score = 0
audio_emotion_label = "Neutral"

# data_columns = ["Timestamp", "Audio Depression Score", "Audio Emotion", 
#                 "Face Depression Score", "Face Depression Stage", 
#                 "Euler Angle Yaw", "Euler Angle Pitch", "Euler Angle Roll"]
# session_data = pd.DataFrame(columns=data_columns)

# Streamlit UI setup
st.title("Real-Time Depression Analysis")
st.sidebar.title("Controls")
start_button = st.sidebar.button("Start Live Analysis")
stop_button = st.sidebar.button("Stop Live Analysis")
record_audio = st.sidebar.checkbox("Record Audio", value=True)

# Function to get head pose
def get_head_pose(shape):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),   # Nose tip
        (shape.part(8).x, shape.part(8).y),     # Chin
        (shape.part(36).x, shape.part(36).y),   # Left eye left corner
        (shape.part(45).x, shape.part(45).y),   # Right eye right corner
        (shape.part(48).x, shape.part(48).y),   # Left mouth corner
        (shape.part(54).x, shape.part(54).y)    # Right mouth corner
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))  # No lens distortion
    _, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    return rotation_matrix

# Function for emotion analysis
def analyze_emotion(frame):
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(results, list):
            results = results[0]
        return results
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return None

# Function to extract features and predict
def extract_features_and_predict(audio_data):
    global audio_depression_score, audio_emotion_label
    WAVE_OUTPUT_FILENAME = "live_audio.wav"
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_data))
    
    audio, sample_rate = sf.read(WAVE_OUTPUT_FILENAME)
    features = smile.process_signal(audio, sample_rate)

    features_flat = features.mean().values.flatten().tolist()
    features_df = pd.DataFrame([features_flat])
    predicted_label = clf.predict(features_df)

    audio_emotion_label = predicted_label[0]
    audio_depression_score = calculate_depression_score_from_audio(features_flat)
    
    os.remove(WAVE_OUTPUT_FILENAME)

def calculate_depression_score_from_audio(features_flat):
    low_energy = features_flat[0] < -10
    low_variance = features_flat[1] < 0.5
    high_negativity = features_flat[-1] < -20

    score = 0
    if low_energy:
        score += 0.3
    if low_variance:
        score += 0.4
    if high_negativity:
        score += 0.3
    
    return score * 100  # Percentage-based depression score

def calculate_depression_score_from_face(emotions):
    depression_weights = {
        'sad': 1.0,
        'angry': 0.8,
        'neutral': 0.4,
        'fear': 0.7,
        'happy': -0.5
    }
    score = sum(depression_weights.get(emotion, 0) * emotions[emotion] for emotion in emotions)
    return max(0, min(score, 100))  # Constrain score to [0, 100]

def get_depression_stage(score):
    if score < 30:
        return "Stage 1: Low"
    elif 30 <= score < 70:
        return "Stage 2: Moderate"
    else:
        return "Stage 3: High"

# Convert rotation matrix to Euler angles
def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = 0
    
    return np.degrees(x_angle), np.degrees(y_angle), np.degrees(z_angle)

def save_dataframe(dataframe):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"session_data_{timestamp}.csv"
    dataframe.to_csv(filename, index=False)
    st.success(f"Data saved as {filename}")

# Initialize session_data in Streamlit's session state if it doesn't exist
# Initialize session_data in Streamlit's session state if it doesn't exist
if 'session_data' not in st.session_state:
    data_columns = [
        "Timestamp", "Audio Depression Score", "Audio Emotion Label",
        "Face Depression Score", "Face Depression Stage", "Euler_Yaw",
        "Euler_Pitch", "Euler_Roll"  # Use consistent naming with underscores
    ]
    st.session_state.session_data = pd.DataFrame(columns=data_columns)


# Function for live audio processing
def live_audio_processing():
    audio_data = []
    frame_count = 0
    frames_per_segment = int(RATE * FRAME_DURATION / CHUNK)

    while True:
        data = stream.read(CHUNK)
        audio_data.append(data)
        frame_count += 1

        if frame_count >= frames_per_segment:
            extract_features_and_predict(audio_data)
            audio_data = []  # Reset for the next segment
            frame_count = 0
            time.sleep(0.5)


# import streamlit as st
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to display the summary report
def display_summary_report():
    st.header("Summary Report and Analysis")
    st.dataframe(st.session_state.session_data)
    # Summarize key metrics
    summary = {
        'Average Audio Depression Score': st.session_state.session_data['Audio Depression Score'].mean(),
        'Average Face Depression Score': st.session_state.session_data['Face Depression Score'].mean(),
        'Most Common Audio Emotion': st.session_state.session_data['Audio Emotion Label'].mode()[0],
        'Most Common Face Depression Stage': st.session_state.session_data['Face Depression Stage'].mode()[0]
    }
    
    # Display summary statistics
    st.subheader("Summarized Metrics")
    for key, value in summary.items():
        st.write(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    # Display summarized session data as a single row
    summarized_row = st.session_state.session_data.mean(numeric_only=True).to_frame().T
    st.subheader("Summarized Data Row")
    st.dataframe(summarized_row)

    # Plot Audio and Face Depression Scores over time
    st.subheader("Depression Scores Over Time")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.lineplot(data=st.session_state.session_data, x="Timestamp", y="Audio Depression Score", ax=ax[0], marker="o", color="b")
    ax[0].set_title("Audio Depression Score Over Time")
    ax[0].tick_params(axis='x', rotation=45)

    sns.lineplot(data=st.session_state.session_data, x="Timestamp", y="Face Depression Score", ax=ax[1], marker="o", color="r")
    ax[1].set_title("Face Depression Score Over Time")
    ax[1].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

    # Euler Angles as Bar Chart
    # Euler Angles as Bar Chart
    st.subheader("Euler Angles (Yaw, Pitch, Roll) Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    st.session_state.session_data[['Timestamp', 'Euler Yaw', 'Euler Pitch', 'Euler Roll']].set_index('Timestamp').plot(kind='bar', ax=ax)
    ax.set_title("Euler Angles Over Time")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)



# Initialize audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Start audio processing in a separate thread
audio_thread = threading.Thread(target=live_audio_processing)
audio_thread.daemon = True

# Start live analysis if the start button is clicked
if start_button:
    st.write("Starting live analysis...")
    audio_thread.start()
    
    cap = cv2.VideoCapture(0)
    video_placeholder = st.empty()  # Placeholder for video stream

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            rotation_matrix = get_head_pose(shape)

            # Draw face bounding box
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

            # Analyze emotion
            face_roi = frame[face.top():face.bottom(), face.left():face.right()]
            emotion_result = analyze_emotion(face_roi)

            if emotion_result is not None:
                face_depression_score = calculate_depression_score_from_face(emotion_result['emotion'])
                face_depression_stage = get_depression_stage(face_depression_score)

                cv2.putText(frame, f"Depression Stage (Face): {face_depression_stage} ({face_depression_score:.2f}%)",
                            (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) 

                euler_angles = rotation_matrix_to_euler_angles(rotation_matrix)
                euler_yaw, euler_pitch, euler_roll = euler_angles

                # Append data to session_data DataFrame
                st.session_state.session_data.loc[len(st.session_state.session_data)] = [
            time.strftime("%Y-%m-%d %H:%M:%S"), 
            audio_depression_score,
            audio_emotion_label, 
            face_depression_score,
            face_depression_stage, 
            euler_yaw, 
            euler_pitch, 
            euler_roll
        ]

                # Display the analysis results
                st.write("Session Data:")
                st.dataframe(st.session_state.session_data)
                st.write(f"Audio Depression Score: {audio_depression_score:.2f}%")
                st.write(f"Audio Emotion: {audio_emotion_label}")
                st.write(f"Face Depression Score: {face_depression_score:.2f}%")
                st.write(f"Face Depression Stage: {face_depression_stage}")
                st.write(f"Euler Angles (Yaw, Pitch, Roll): ({euler_yaw:.2f}, {euler_pitch:.2f}, {euler_roll:.2f})")

                cv2.putText(frame, f"Yaw: {euler_angles[0]:.2f}, Pitch: {euler_angles[1]:.2f}, Roll: {euler_angles[2]:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the audio-based depression score and emotion
        cv2.putText(frame, f"Audio Depression Stage: {get_depression_stage(audio_depression_score)} ({audio_depression_score:.2f}%)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f"Audio Emotion: {audio_emotion_label}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display the video stream
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        
        if stop_button:
            # st.dataframe(st.session_state.session_data)
            break

    # Release video capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Display summary report after stopping analysis
if stop_button:
    # st.write("Live analysis stopped.")
    # st.dataframe(st.session_state.session_data)
    csv = save_dataframe(st.session_state.session_data)  # Save session data to CSV
    display_summary_report()
    # st.download_button(
    #     label="Download Session Data as CSV",
    #     data=csv,
    #     file_name='session_data.csv',
    #     mime='text/csv'
    # )
st.sidebar.write("Note: Ensure the camera and microphone are accessible.")