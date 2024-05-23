# src/transformer_model.py

import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Define file paths
raw_data_path = 'data/raw/customer_queries.csv'
processed_data_path = 'data/processed/processed_data.pkl'
transformer_model_path = 'models/transformer_model.h5'

# Create directories if they don't exist
os.makedirs(os.path.dirname(transformer_model_path), exist_ok=True)

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

# Define Transformer model components
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, _ = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Build Transformer model
def build_transformer_model(vocab_size, max_sequence_length):
    embed_dim = 256
    num_heads = 8
    ff_dim = 512

    inputs = Input(shape=(max_sequence_length,))
    embedding_layer = TokenAndPositionEmbedding(max_sequence_length, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Build and train the Transformer model
model = build_transformer_model(vocab_size, max_sequence_length)
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(transformer_model_path, save_best_only=True, monitor='val_loss')

print("Training the model...")
history = model.fit(
    X_train, y_train_one_hot,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Save the trained model
model.save(transformer_model_path)
print(f"Model saved at {transformer_model_path}")

print("Transformer model building and training completed!")
