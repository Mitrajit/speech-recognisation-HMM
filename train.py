import os
import numpy as np
import librosa
from hmmlearn import hmm
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    X = []
    y = []
    for digit in range(10):
        digit_dir = os.path.join(data_dir, str(digit))
        for file in os.listdir(digit_dir):
            if file.endswith(".wav"):
                audio_file = os.path.join(digit_dir, file)
                mfcc = extract_features(audio_file)
                X.append(mfcc.T)
                y.append(digit)
    return X, y


def extract_features(audio_file, max_pad_len=40):
    signal, sample_rate = librosa.load(audio_file)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate)
    if (max_pad_len > mfcc.shape[1]):
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    return mfcc


# Set the directory where your data is located
data_dir = 'data'

# Load the data
X, y = load_data(data_dir)

# Number of states in HMM, and number of Gaussians per state
n_components = 5
models = [hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000) for _ in range(10)]

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train each HMM
for digit in range(10):
    X_digit = np.concatenate([X_train[i] for i in range(len(y_train)) if y_train[i] == digit])
    lengths = [X_train[i].shape[0] for i in range(len(y_train)) if y_train[i] == digit]
    models[digit].fit(X_digit, lengths=lengths)

# Predictions
# for x, true_label in zip(X_test, y_test):
#     scores = [model.score(x) for model in models]
#     predicted_digit = np.argmax(scores)
#     print(f"True label: {true_label}, Predicted: {predicted_digit}")

# Predictions
correct_predictions = 0
total_predictions = 0

for x, true_label in zip(X_test, y_test):
    scores = [model.score(x) for model in models]
    predicted_digit = np.argmax(scores)
    
    if predicted_digit == true_label:
        correct_predictions += 1
    total_predictions += 1

# Calculate the accuracy in percentage
accuracy = (correct_predictions / total_predictions) * 100
print(f'Accuracy: {accuracy:.2f}%')

from joblib import dump

# Specify the directory where you want to save the models
models_dir = 'models'  # You can change this to your preferred directory

# Create the directory if it doesn't exist
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save the models
for digit, model in enumerate(models):
    filename = os.path.join(models_dir, f"model_for_digit_{digit}.joblib")
    dump(model, filename)