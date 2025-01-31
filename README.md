# Emotional Analysis

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/emotional-analysis.git
   ```
2. Navigate to the project directory:
   ```
   cd emotional-analysis
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download the necessary models:
   - Download the `shape_predictor_68_face_landmarks.dat` file from the [dlib website](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract it to the project directory.
   - Download the `haarcascade_frontalface_default.xml` file from the [OpenCV repository](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and place it in the project directory.

## Usage

1. Run the Streamlit application:
   ```
   streamlit run new_cam_audio_streamlit.py
   ```
2. The application will start and display a video stream with real-time depression analysis.
3. The application will save the session data to a CSV file, which can be accessed by clicking the "Save Session Data" button.
4. The "Generate Summary Report" button will display a summary of the session data, including visualizations.

## API

The project includes the following Python files:

1. `audio_ml.py`: This file trains a RandomForestClassifier model on the EMO-DB dataset and saves the model to a `.pkl` file.
2. `conver.py`: This file extracts features from the EMO-DB dataset using OpenSMILE and saves the features and labels to a CSV file.
3. `new_cam_audio_streamlit.py`: This is the main Streamlit application that performs real-time depression analysis using the webcam.

## Contributing

Contributions to this project are welcome. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

The project does not currently include any automated tests. However, you can manually test the application by running the Streamlit application and verifying the functionality.