# src/exploratory_data_analysis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.probability import FreqDist

# Define file paths
raw_data_path = 'data/raw/customer_queries.csv'
processed_data_path = 'data/processed/processed_data.pkl'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(figures_path, exist_ok=True)

# Load raw data
print("Loading raw data...")
data = pd.read_csv(raw_data_path)

# Display the first few rows of the dataset
print("Raw Data:")
print(data.head())

# Load processed data
print("Loading processed data...")
with open(processed_data_path, 'rb') as f:
    processed_data = pickle.load(f)

# Display the processed data keys
print("Processed Data Keys:")
print(processed_data.keys())

# Summary statistics
print("Summary Statistics:")
print(data.describe())

# Distribution of question lengths
question_lengths = data['question'].apply(lambda x: len(x.split()))
answer_lengths = data['answer'].apply(lambda x: len(x.split()))

plt.figure(figsize=(12, 6))
sns.histplot(question_lengths, bins=50, kde=True, color='blue', label='Questions')
sns.histplot(answer_lengths, bins=50, kde=True, color='green', label='Answers')
plt.title('Distribution of Question and Answer Lengths')
plt.xlabel('Length (number of words)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(figures_path, 'question_answer_lengths_distribution.png'))
plt.show()

# Most common words in questions and answers
all_questions = ' '.join(data['question_clean'].tolist())
all_answers = ' '.join(data['answer_clean'].tolist())

questions_freq_dist = FreqDist(nltk.word_tokenize(all_questions))
answers_freq_dist = FreqDist(nltk.word_tokenize(all_answers))

# Display most common words
print("Most Common Words in Questions:")
print(questions_freq_dist.most_common(10))

print("Most Common Words in Answers:")
print(answers_freq_dist.most_common(10))

# Plot most common words
questions_common_words = pd.DataFrame(questions_freq_dist.most_common(20), columns=['Word', 'Frequency'])
answers_common_words = pd.DataFrame(answers_freq_dist.most_common(20), columns=['Word', 'Frequency'])

plt.figure(figsize=(12, 6))
sns.barplot(x='Word', y='Frequency', data=questions_common_words, color='blue')
plt.title('Most Common Words in Questions')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.savefig(os.path.join(figures_path, 'common_words_questions.png'))
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Word', y='Frequency', data=answers_common_words, color='green')
plt.title('Most Common Words in Answers')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.savefig(os.path.join(figures_path, 'common_words_answers.png'))
plt.show()

# Word clouds for questions and answers
questions_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_questions)
answers_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_answers)

plt.figure(figsize=(12, 6))
plt.imshow(questions_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Questions')
plt.axis('off')
plt.savefig(os.path.join(figures_path, 'wordcloud_questions.png'))
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(answers_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Answers')
plt.axis('off')
plt.savefig(os.path.join(figures_path, 'wordcloud_answers.png'))
plt.show()

# Distribution of vocabulary size
vocab_size = len(processed_data['tokenizer'].word_index) + 1
print(f"Vocabulary Size: {vocab_size}")

# Plot the distribution of sequence lengths after padding
sequence_lengths = [len(seq) for seq in processed_data['X_train']]
plt.figure(figsize=(12, 6))
sns.histplot(sequence_lengths, bins=50, kde=True, color='purple')
plt.title('Distribution of Padded Sequence Lengths')
plt.xlabel('Length (number of tokens)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(figures_path, 'sequence_lengths_distribution.png'))
plt.show()

print("Exploratory Data Analysis completed!")
