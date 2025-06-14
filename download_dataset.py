import kagglehub
import os
import shutil
from pathlib import Path

def download_and_organize_dataset():
    # Download the dataset
    print("Downloading dataset...")
    dataset_path = kagglehub.dataset_download("mohammedabdeldayem/the-fake-or-real-dataset")
    print(f"Dataset downloaded to: {dataset_path}")

    # Create directories if they don't exist
    real_audio_dir = Path("real_audio")
    deepfake_audio_dir = Path("deepfake_audio")
    real_audio_dir.mkdir(exist_ok=True)
    deepfake_audio_dir.mkdir(exist_ok=True)

    # Clear existing files in the directories
    for file in real_audio_dir.glob("*.wav"):
        file.unlink()
    for file in deepfake_audio_dir.glob("*.wav"):
        file.unlink()

    # Organize files
    print("Organizing files...")
    dataset_path = Path(dataset_path)
    
    # Process each sub-dataset
    for subdir in dataset_path.iterdir():
        if subdir.is_dir():
            print(f"Processing {subdir.name}...")
            
            # Process real audio files
            real_files = list(subdir.glob("**/real/*.wav"))
            for file in real_files:
                dest = real_audio_dir / f"{subdir.name}_{file.name}"
                shutil.copy2(file, dest)
                print(f"Copied {file.name} to real_audio directory")
            
            # Process fake audio files
            fake_files = list(subdir.glob("**/fake/*.wav"))
            for file in fake_files:
                dest = deepfake_audio_dir / f"{subdir.name}_{file.name}"
                shutil.copy2(file, dest)
                print(f"Copied {file.name} to deepfake_audio directory")

    print("\nDataset organization complete!")
    print(f"Real audio files: {len(list(real_audio_dir.glob('*.wav')))}")
    print(f"Deepfake audio files: {len(list(deepfake_audio_dir.glob('*.wav')))}")

if __name__ == "__main__":
    download_and_organize_dataset() 