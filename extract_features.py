import os
import glob
import librosa
import numpy as np
import pandas as pd

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_features = np.mean(mfccs.T, axis=0)
    
    # Create a DataFrame with MFCC features
    feature_names = [f'MFCC_{i+1}' for i in range(n_mfcc)]
    df = pd.DataFrame([mfcc_features], columns=feature_names)
    
    # Add filename and label columns
    df['filename'] = os.path.basename(audio_path)
    
    return df

def process_directory(directory, label):
    print(f"Processing directory: {directory}")
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    all_features = []
    
    for audio_path in audio_files:
        print(f"Processing: {os.path.basename(audio_path)}")
        df = extract_mfcc_features(audio_path)
        if df is not None:
            df['label'] = label
            all_features.append(df)
    
    return pd.concat(all_features, ignore_index=True) if all_features else None

def main():
    genuine_dir = "real_audio"
    deepfake_dir = "deepfake_audio"
    
    # Process genuine audio files
    genuine_features = process_directory(genuine_dir, label=0)
    
    # Process deepfake audio files
    deepfake_features = process_directory(deepfake_dir, label=1)
    
    # Combine features
    if genuine_features is not None and deepfake_features is not None:
        all_features = pd.concat([genuine_features, deepfake_features], ignore_index=True)
        
        # Save to CSV
        output_file = 'mfcc_features.csv'
        all_features.to_csv(output_file, index=False)
        print(f"\nFeatures saved to {output_file}")
        print(f"Total samples processed: {len(all_features)}")
        print(f"Genuine samples: {len(genuine_features)}")
        print(f"Deepfake samples: {len(deepfake_features)}")
    else:
        print("Error: No features were extracted from one or both directories")

if __name__ == "__main__":
    main() 