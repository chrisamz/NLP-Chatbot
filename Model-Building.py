# src/model_building.py

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define file paths
feature_engineered_data_path = 'data/processed/feature_engineered_data.pkl'
seq2seq_model_path = 'models/seq2seq_model.h5'

# Create directories if they don't exist
os.makedirs(os.path.dirname(seq2seq_model_path), exist_ok=True)

# Load feature-engineered data
print("Loading feature-engineered data...")
with open(feature_engineered_data_path, 'rb') as f:
    feature_engineered_data = pickle.load(f)

# Extract data from the feature_engineered_data dictionary
X_train = feature_engineered_data['X_train']
X_test = feature_engineered_data['X_test']
y_train = feature_engineered_data['y_train']
y_test = feature_engineered_data['y_test']
tokenizer = feature_engineered_data['tokenizer']
max_sequence_length = feature_engineered_data['max_sequence_length']

# Display the shape of the data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Build the Seq2Seq model
def build_seq2seq_model(vocab_size, max_sequence_length):
    # Define the encoder
    encoder_inputs = Input(shape=(max_sequence_length,))
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Define the decoder
    decoder_inputs = Input(shape=(max_sequence_length,))
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense
# Define the model that will turn encoder_inputs and decoder_inputs into decoder_outputs
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build the model
vocab_size = len(tokenizer.word_index) + 1
model = build_seq2seq_model(vocab_size, max_sequence_length)

# Display the model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(seq2seq_model_path, save_best_only=True, monitor='val_loss')

# Train the model
print("Training the model...")
history = model.fit(
    [X_train, y_train[:, :-1]],
    y_train[:, 1:, :],
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate([X_test, y_test[:, :-1]], y_test[:, 1:])
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Save the trained model
model.save(seq2seq_model_path)
print(f"Model saved at {seq2seq_model_path}")

print("Model building and training completed!")
