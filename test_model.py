import os
import glob
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def load_features_from_csv(csv_file='mfcc_features.csv'):
    """Load features from existing CSV file"""
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run extract_features.py first.")
        return None, None
    
    print(f"Loading features from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Clean the data - remove rows with NaN or invalid labels
    df = df.dropna(subset=['label'])
    df = df[df['label'].isin([0, 1])]  # Only keep valid labels
    
    # Extract features and labels
    feature_columns = [col for col in df.columns if col.startswith('MFCC_')]
    X = df[feature_columns].values
    y = df['label'].values.astype(int)  # Convert to int to fix the bincount error
    
    print(f"Loaded {len(X)} samples with {len(feature_columns)} features each")
    print(f"Labels: {np.bincount(y)} (0: Genuine, 1: Deepfake)")
    
    return X, y

def test_model():
    # Load the saved model and scaler
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"
    
    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
        print("Error: Model or scaler files not found. Please train the model first using main.py")
        return
    
    print("Loading saved model and scaler...")
    svm_classifier = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    
    # Load features from CSV
    X, y = load_features_from_csv()
    if X is None:
        return
    
    # Split into train and test sets (same random_state as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Test set size: {X_test.shape}")
    print(f"Training set size: {X_train.shape}")
    
    # Scale the test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("Making predictions...")
    y_pred = svm_classifier.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL TESTING RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nConfusion Matrix:")
    print(confusion_mtx)
    
    # Calculate additional metrics
    tn, fp, fn, tp = confusion_mtx.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    print(f"\nTrue Negatives (Genuine correctly classified): {tn}")
    print(f"False Positives (Genuine misclassified as Deepfake): {fp}")
    print(f"False Negatives (Deepfake misclassified as Genuine): {fn}")
    print(f"True Positives (Deepfake correctly classified): {tp}")

if __name__ == "__main__":
    test_model() 