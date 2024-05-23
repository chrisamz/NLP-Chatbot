# src/seq2seq_model.py

import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

# Define file paths
raw_data_path = 'data/raw/customer_queries.csv'
processed_data_path = 'data/processed/processed_data.pkl'
seq2seq_model_path = 'models/seq2seq_model.h5'

# Create directories if they don't exist
os.makedirs(os.path.dirname(seq2seq_model_path), exist_ok=True)

# Load raw data
print("Loading raw data...")
data = pd.read_csv(raw_data_path)

# Data Preprocessing
print("Cleaning and preprocessing data...")

# Define a function for text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Apply text cleaning to the questions and answers
data['question_clean'] = data['question'].apply(clean_text)
data['answer_clean'] = data['answer'].apply(clean_text)

# Tokenization and Padding
print("Tokenizing and padding data...")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['question_clean'].tolist() + data['answer_clean'].tolist())
vocab_size = len(tokenizer.word_index) + 1

question_sequences = tokenizer.texts_to_sequences(data['question_clean'].tolist())
answer_sequences = tokenizer.texts_to_sequences(data['answer_clean'].tolist())

max_sequence_length = max([len(seq) for seq in question_sequences + answer_sequences])
question_padded = pad_sequences(question_sequences, maxlen=max_sequence_length, padding='post')
answer_padded = pad_sequences(answer_sequences, maxlen=max_sequence_length, padding='post')

X_train, X_test, y_train, y_test = train_test_split(question_padded, answer_padded, test_size=0.2, random_state=42)

# One-hot encode the output sequences
print("One-hot encoding output sequences...")

def one_hot_encode(sequences, vocab_size):
    one_hot_encoded = np.zeros((len(sequences), max_sequence_length, vocab_size), dtype='float32')
    for i, seq in enumerate(sequences):
        for t, word_id in enumerate(seq):
            if word_id != 0:
                one_hot_encoded[i, t, word_id] = 1.0
    return one_hot_encoded

y_train_one_hot = one_hot_encode(y_train, vocab_size)
y_test_one_hot = one_hot_encode(y_test, vocab_size)

# Save processed data
print("Saving processed data...")
processed_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train_one_hot,
    'y_test': y_test_one_hot,
    'tokenizer': tokenizer,
    'max_sequence_length': max_sequence_length
}
with open(processed_data_path, 'wb') as f:
    pickle.dump(processed_data, f)

# Build the Seq2Seq model
def build_seq2seq_model(vocab_size, max_sequence_length):
    encoder_inputs = Input(shape=(max_sequence_length,))
    encoder_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(encoder_inputs)
    encoder_lstm = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(max_sequence_length,))
    decoder_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and train the Seq2Seq model
model = build_seq2seq_model(vocab_size, max_sequence_length)
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(seq2seq_model_path, save_best_only=True, monitor='val_loss')

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

print("Seq2Seq model building and training completed!")
