# src/dialogue_management.py

import os
import openai
import pandas as pd
import pickle
from flask import Flask, request, jsonify

# Define file paths
fine_tuned_model_path = 'models/gpt3_fine_tuned_model.pkl'
api_key_path = 'config/openai_api_key.txt'

# Load OpenAI API key
with open(api_key_path, 'r') as file:
    openai_api_key = file.read().strip()

openai.api_key = openai_api_key

# Load the fine-tuned GPT-3 model ID
with open(fine_tuned_model_path, 'rb') as f:
    fine_tuned_model_id = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# In-memory storage for conversation context
conversation_context = {}

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

# Function to manage dialogue and maintain context
def manage_dialogue(user_id, user_input):
    if user_id not in conversation_context:
        conversation_context[user_id] = []

    # Append the user input to the context
    conversation_context[user_id].append(f"User: {user_input}")

    # Construct the prompt from the context
    prompt = "\n".join(conversation_context[user_id]) + "\nAI:"

    # Generate the response
    response = generate_response(prompt, fine_tuned_model_id)

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
