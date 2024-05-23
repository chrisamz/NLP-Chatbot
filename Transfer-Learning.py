# src/transfer_learning_gpt3.py

import os
import openai
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split

# Define file paths
raw_data_path = 'data/raw/customer_queries.csv'
processed_data_path = 'data/processed/processed_data.pkl'
fine_tuned_model_path = 'models/gpt3_fine_tuned_model.pkl'
api_key_path = 'config/openai_api_key.txt'

# Create directories if they don't exist
os.makedirs(os.path.dirname(fine_tuned_model_path), exist_ok=True)

# Load OpenAI API key
with open(api_key_path, 'r') as file:
    openai_api_key = file.read().strip()

openai.api_key = openai_api_key

# Load raw data
print("Loading raw data...")
data = pd.read_csv(raw_data_path)

# Display the first few rows of the dataset
print("Raw Data:")
print(data.head())

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Prepare data for fine-tuning
def prepare_data(data):
    prepared_data = []
    for index, row in data.iterrows():
        prepared_data.append({
            "prompt": row['question_clean'] + "\n\n###\n\n",
            "completion": row['answer_clean'] + "\n"
        })
    return prepared_data

train_prepared_data = prepare_data(train_data)
val_prepared_data = prepare_data(val_data)

# Save prepared data to JSONL files
train_jsonl_path = 'data/processed/train_prepared_data.jsonl'
val_jsonl_path = 'data/processed/val_prepared_data.jsonl'

with open(train_jsonl_path, 'w') as f:
    for item in train_prepared_data:
        f.write(json.dumps(item) + "\n")

with open(val_jsonl_path, 'w') as f:
    for item in val_prepared_data:
        f.write(json.dumps(item) + "\n")

# Fine-tune the GPT-3 model
print("Fine-tuning the GPT-3 model...")
fine_tuning_response = openai.FineTune.create(
    training_file=openai.File.create(file=open(train_jsonl_path, "rb"), purpose='fine-tune').id,
    validation_file=openai.File.create(file=open(val_jsonl_path, "rb"), purpose='fine-tune').id,
    model="davinci",
    n_epochs=4
)

# Retrieve the fine-tuned model ID
fine_tuned_model_id = fine_tuning_response['fine_tuned_model']

# Save the fine-tuned model ID
with open(fine_tuned_model_path, 'wb') as f:
    pickle.dump(fine_tuned_model_id, f)

print(f"Fine-tuned model ID saved at {fine_tuned_model_path}")

# Function to generate a response using the fine-tuned GPT-3 model
def generate_response(prompt, model_id):
    response = openai.Completion.create(
        model=model_id,
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        n=1,
        stop=["\n"]
    )
    return response.choices[0].text.strip()

# Test the fine-tuned model
test_prompt = "How can I reset my password?\n\n###\n\n"
response = generate_response(test_prompt, fine_tuned_model_id)
print(f"Test Prompt: {test_prompt}")
print(f"Response: {response}")

print("Transfer learning with GPT-3 completed!")
