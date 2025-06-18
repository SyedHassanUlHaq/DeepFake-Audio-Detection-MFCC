import os
import librosa
import numpy as np

def is_silent(file_path, threshold=0.01):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Calculate the root mean square (RMS) value
    rms = np.sqrt(np.mean(y**2))

    # Check if the RMS value is below the threshold
    return rms < threshold

def delete_silent_files(directory, threshold=0.01):
    deleted_count = 0

    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".wav") or filename.endswith(".mp3"):  # Add other audio formats if needed
            file_path = os.path.join(directory, filename)

            # Check if the file is silent
            if is_silent(file_path, threshold):
                print(f"Deleting silent file: {filename}")
                os.remove(file_path)
                deleted_count += 1

    return deleted_count

# Directories containing audio files
directories = ['deepfake_audio', 'real_audio']

# Delete silent files in each directory and count the number of deletions
total_deleted = 0
for directory in directories:
    if os.path.isdir(directory):
        print(f"Processing directory: {directory}")
        deleted = delete_silent_files(directory)
        total_deleted += deleted
        print(f"Number of files deleted in {directory}: {deleted}")
    else:
        print(f"Directory not found: {directory}")

print(f"Total number of silent audio files deleted: {total_deleted}")
