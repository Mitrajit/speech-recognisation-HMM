from joblib import load
import librosa
import numpy as np

# Load the models
models = [load(f"models/model_for_digit_{i}.joblib") for i in range(10)]

# Function to extract features, you should use the same function you used for training
def extract_features(audio_file, max_pad_len=40):
    signal, sample_rate = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate)
    if (max_pad_len > mfcc.shape[1]):
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc

# Live audio recognition
def recognize_live_audio(audio_file):
    # Extract MFCC features
    mfcc_features = extract_features(audio_file).T
    
    # Score the features using the HMM models
    scores = [model.score(mfcc_features) for model in models]
    
    # The digit is the model with the highest score
    predicted_digit = np.argmax(scores)
    
    return predicted_digit

# Example usage:
audio_file = "data/5/5_02_17.wav"
recognized_digit = recognize_live_audio(audio_file)
print(f"Recognized digit: {recognized_digit}")
