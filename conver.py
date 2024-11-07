import opensmile
import soundfile as sf
import pandas as pd
import os
from datasets import load_dataset

# Initialize OpenSMILE for extracting features
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

# Load the EMO-DB dataset
dataset = load_dataset("confit/emodb-demo")

# Function to extract features from each audio file
def extract_features_from_audio(audio_array, sample_rate):
    # Extract features using OpenSMILE
    features = smile.process_signal(audio_array, sample_rate)
    return features

# Initialize an empty list to store features and labels
features_list = []
labels_list = []

# Iterate through the dataset and extract features for each sample
for sample in dataset['train']:
    if 'audio' in sample:  # Ensure 'audio' key exists in the dataset
        audio_file_path = sample['audio']['path']
        audio_array = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        label = sample['emotion']  # The emotion label

        # Extract features for the audio
        features = extract_features_from_audio(audio_array, sample_rate)

        if features is not None:  # Ensure features were successfully extracted
            # Flatten the features into a single row and add to the list
            features_flat = features.mean().values.flatten().tolist()  # Taking the mean for simplicity
            
            # Append the corresponding label if it is not 'happiness'
            if label != 'happiness':
                features_list.append(features_flat)
                labels_list.append(label)
        else:
            print(f"Skipping file: {audio_file_path} due to extraction issue.")
    else:
        print(f"Key 'audio' not found in sample: {sample}")

# Create a DataFrame with the features and labels
features_df = pd.DataFrame(features_list)
features_df['label'] = labels_list

# Save the features and labels to a CSV file
csv_output_path = 'emodb_features.csv'
features_df.to_csv(csv_output_path, index=False)
print(f"Features saved to {csv_output_path}")
