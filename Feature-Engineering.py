# src/feature_engineering.py

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define file paths
processed_data_path = 'data/processed/processed_data.pkl'
feature_engineered_data_path = 'data/processed/feature_engineered_data.pkl'

# Create directories if they don't exist
os.makedirs(os.path.dirname(feature_engineered_data_path), exist_ok=True)

# Load processed data
print("Loading processed data...")
with open(processed_data_path, 'rb') as f:
    processed_data = pickle.load(f)

# Extract data from the processed_data dictionary
X_train = processed_data['X_train']
X_test = processed_data['X_test']
y_train = processed_data['y_train']
y_test = processed_data['y_test']
tokenizer = processed_data['tokenizer']
max_sequence_length = processed_data['max_sequence_length']

# Display the first few rows of the data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# One-hot encode the output sequences
print("One-hot encoding output sequences...")
vocab_size = len(tokenizer.word_index) + 1
encoder = OneHotEncoder(handle_unknown='ignore')

def one_hot_encode(sequences, vocab_size):
    one_hot_encoded = np.zeros((len(sequences), max_sequence_length, vocab_size), dtype='float32')
    for i, seq in enumerate(sequences):
        for t, word_id in enumerate(seq):
            if word_id != 0:  # Skip padding
                one_hot_encoded[i, t, word_id] = 1.0
    return one_hot_encoded

y_train_one_hot = one_hot_encode(y_train, vocab_size)
y_test_one_hot = one_hot_encode(y_test, vocab_size)

# Save feature-engineered data
print("Saving feature-engineered data...")
feature_engineered_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train_one_hot,
    'y_test': y_test_one_hot,
    'tokenizer': tokenizer,
    'max_sequence_length': max_sequence_length
}
with open(feature_engineered_data_path, 'wb') as f:
    pickle.dump(feature_engineered_data, f)

print("Feature engineering completed!")
