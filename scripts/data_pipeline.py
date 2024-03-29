# Data Pipeline Script

# Import necessary libraries
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import pickle

# Define variables imported from Makefile
INPUT_FOLDER_VIDEOS = os.environ.get("INPUT_FOLDER_VIDEOS", "C:/data/videos/seizure C:/data/videos/normal")
OUTPUT_FOLDER_VIDEOS = os.environ.get("OUTPUT_FOLDER_VIDEOS", "C:/data/videos/seizure C:/data/videos/normal")
INPUT_FOLDER_DATA = os.environ.get("INPUT_FOLDER_DATA", "C:/data/videos/seizure C:/data/videos/normal")
OUTPUT_FOLDER_DATA = os.environ.get("OUTPUT_FOLDER_DATA", "C:/data/CSV/seizure C:/data/CSV/normal")
INPUT_CSV_FOLDER = os.environ.get("INPUT_CSV_FOLDER", "C:/data/CSV/seizure C:/data/CSV/normal")
OUTPUT_PICKLE_FOLDER = "C:/data/pickle"

# Define variables
random_seed = 42
n_time_steps = 75
step = 32
n_classes = 2

# Function for data extraction
def extract(input_folder_videos, output_folder_videos, input_folder_data, output_folder_data):
    # Process videos: convert to 25 FPS
    for folder in input_folder_videos.split():
        for filename in os.listdir(folder):
            if filename.endswith(".mp4"):
                input_path = os.path.join(folder, filename)
                output_path = os.path.join(output_folder_videos, os.path.basename(folder), filename)

                video_clip = VideoFileClip(input_path)
                video_clip = video_clip.set_fps(25)
                video_clip.write_videofile(output_path, codec="libx264", fps=25)
                print(f"Processed video: {input_path}")

    print("All videos processed.")

    # Process data extraction

    mp_holistic = mp.solutions.holistic
    # Define the indices of the 21 face landmarks
    face_landmark_indices = [0, 4, 17, 48, 50, 61, 122, 130, 133, 145, 159, 206, 280, 289, 292, 351, 362, 359, 374, 386, 426]

    for folder in input_folder_data.split():
        file_counter = 1  # Counter for naming CSV files
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    output_folder_category = os.path.join(output_folder_data, os.path.basename(folder))
                    os.makedirs(output_folder_category, exist_ok=True)

                    cap = cv2.VideoCapture(video_path)

                    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                        landmark_data = []

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_landmarks = []
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = holistic.process(rgb_frame)

                            # Extract landmark data
                            if results.pose_landmarks:
                                for landmark in results.pose_landmarks.landmark:
                                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                            else:
                                # Fill in missing pose landmarks with 0 values
                                frame_landmarks.extend([0.0] * (33 * 4))  # 33 landmarks with x, y, z, and visibility
                            
                            if results.face_landmarks:
                                for i in face_landmark_indices:
                                    landmark = results.face_landmarks.landmark[i]
                                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                            else:
                                # Fill in missing face landmarks with 0 values
                                frame_landmarks.extend([0.0] * (len(face_landmark_indices) * 3))  # Only for the landmarks in face_landmark_indices

                            if results.left_hand_landmarks:
                                for landmark in results.left_hand_landmarks.landmark:
                                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                            else:
                                # Fill in missing left hand landmarks with 0 values
                                frame_landmarks.extend([0.0] * (21 * 3))  # 21 left hand landmarks with x, y, and z

                            if results.right_hand_landmarks:
                                for landmark in results.right_hand_landmarks.landmark:
                                    frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                            else:
                                # Fill in missing right hand landmarks with 0 values
                                frame_landmarks.extend([0.0] * (21 * 3))  # 21 right hand landmarks with x, y, and z

                            # Append the landmark data for the current frame to the landmark_data list
                            landmark_data.append(frame_landmarks)

                        # Define headers
                        pose_headers = []
                        for i in range(33):
                            pose_headers.extend([f'{i}_x_pose', f'{i}_y_pose', f'{i}_z_pose', f'{i}_vis_pose'])
                
                        face_headers = []
                        for i in face_landmark_indices:
                                face_headers.extend([f'{i}_x_face', f'{i}_y_face', f'{i}_z_face'])

                        left_hand_headers = []
                        for i in range(21):
                            left_hand_headers.extend([f'{i}_x_left', f'{i}_y_left', f'{i}_z_left'])

                        right_hand_headers = []
                        for i in range(21):
                            right_hand_headers.extend([f'{i}_x_right', f'{i}_y_right', f'{i}_z_right'])

                        headers = pose_headers + face_headers + left_hand_headers + right_hand_headers

                        df = pd.DataFrame(landmark_data, columns=headers)
                        category = os.path.basename(folder)
                        df.insert(0, "category", category)

                        output_filename = f"{category}{file_counter:02d}.csv"  # Formatted with zero-padded numbers
                        output_csv_path = os.path.join(output_folder_category, output_filename)
                        df.to_csv(output_csv_path, index=False)

                    cap.release()
                    file_counter += 1  # Increment counter for the next file

        print("Data extraction completed.")

    ## Initialize an empty list to store DataFrames
    dfs = []

    for folder in input_folder_data.split():
        # Get a list of CSV files in the folder
        csv_files = [file for file in os.listdir(folder) if file.endswith('.csv')]

        for csv_file in csv_files:
            # Load the CSV file into a DataFrame
            data = pd.read_csv(os.path.join(folder, csv_file))

            # Append the DataFrame to the list
            dfs.append(data)

    ## Concatenate all DataFrames in the list
    df = pd.concat(dfs, ignore_index=True)

    return df

# Function for data balancing
def balance(df):
    ## Apply SMOTE to balance the classes
    smote = SMOTE(sampling_strategy='minority')
    X_sm, y_sm = smote.fit_resample(df.drop('category', axis=1), df['category'])

    ## Create a DataFrame with the rebalanced data
    balanced_df = pd.concat([pd.DataFrame(y_sm, columns=['category']), pd.DataFrame(X_sm)], axis=1)
    
    return balanced_df

def label(balanced_df):
    # Initialize lists to store segments and labels
    segments = []
    labels = []

    # Loop through the data with a specified step
    for i in range(0, balanced_df.shape[0], step):
        mylist = []
        label = None  # Initialize label as None

        if i + n_time_steps < balanced_df.shape[0]:
            for j in range(i, i + n_time_steps):
                mylist.append(balanced_df.iloc[j, 1:322].values)  #  As 321 feature columns

            # Check the label for the segment
            if 'seizure' in balanced_df['category'].iloc[i:i + n_time_steps].values:
                label = 1  # Set label to 1 if 'seizure' is present in this segment
            else:
                label = 0  # Set label to 0 if 'normal' is not present

            segments.append(mylist)
            labels.append(label)

    # Convert segments and labels to NumPy arrays
    reshaped_segments = np.array(segments, dtype=np.float32)
    labels_binary = np.array(labels, dtype=np.float32)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels_binary, test_size=0.2, random_state=random_seed)

    # One-hot encode the labels
    n_classes = 2
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)

    # Save X_train, X_test, y_train, y_test to pickle files
    pickle_folder = os.environ.get("OUTPUT_PICKLE_FOLDER")
    os.makedirs(pickle_folder, exist_ok=True)

    with open(os.path.join(pickle_folder, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)

    with open(os.path.join(pickle_folder, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)

    with open(os.path.join(pickle_folder, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)

    with open(os.path.join(pickle_folder, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)

    return X_train, X_test, y_train, y_test

def main():
    # Run the data preprocessing pipeline
    df = extract(INPUT_FOLDER_VIDEOS, OUTPUT_FOLDER_VIDEOS, INPUT_FOLDER_DATA, OUTPUT_FOLDER_DATA)
    balanced_df = balance(df)
    X_train, X_test, y_train, y_test = label(balanced_df)
    print("Data preprocessing pipeline completed.")

if __name__ == "__main__":
    main()
