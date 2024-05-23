# src/gpt4_chatbot.py

import os
import openai
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split

# Define file paths
raw_data_path = 'data/raw/customer_queries.csv'
api_key_path = 'config/openai_api_key.txt'

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

# Prepare data for fine-tuning (if necessary)
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

# Initialize Flask app
app = Flask(__name__)

# In-memory storage for conversation context
conversation_context = {}

# Function to generate a response using GPT-4
def generate_response(prompt):
    response = openai.Completion.create(
        model="gpt-4",  # Assuming GPT-4 is available as a model identifier
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        top_p=0.9,
        n=1,
        stop=["\n"]
    )
    return response.choices[0].text.strip()

# Function to manage dialogue and maintain context
def manage_dialogue(user_id, user_input):
    if user_id not in conversation_context:
        conversation_context[user_id] = []

    # Append the user input to the context
    conversation_context[user_id].append(f"User: {user_input}")

    # Construct the prompt from the context
    prompt = "\n".join(conversation_context[user_id]) + "\nAI:"

    # Generate the response
    response = generate_response(prompt)

    # Append the AI response to the context
    conversation_context[user_id].append(f"AI: {response}")

    # Return the response
    return response

@app.route('/')
def home():
    return "Customer Service Chatbot API"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the JSON data from the request
        json_data = request.get_json()
        
        # Extract user ID and user input
        user_id = json_data.get('user_id')
        user_input = json_data.get('user_input')
        
        # Generate the response
        response = manage_dialogue(user_id, user_input)
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
