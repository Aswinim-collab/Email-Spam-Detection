"""
ML-focused script to explore the spam email dataset
Pure ML side - no backend API or frontend integration
Uses only pandas and numpy for exploration (no visualization dependencies)
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import json

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('spam_emails.csv')

print("\n=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head(10))

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Class Distribution ===")
class_counts = df['label'].value_counts()
print(class_counts)
print(f"Spam (1): {class_counts[1]} emails")
print(f"Not Spam (0): {class_counts[0]} emails")
print(f"Spam percentage: {class_counts[1]/len(df)*100:.2f}%")

# Text analysis
print("\n=== Text Length Analysis ===")
df['email_length'] = df['email'].str.len()
print(f"Average email length: {df['email_length'].mean():.2f} characters")
print(f"Max email length: {df['email_length'].max()} characters")
print(f"Min email length: {df['email_length'].min()} characters")

# Separate spam and ham for comparison
spam_emails = df[df['label'] == 1]['email']
ham_emails = df[df['label'] == 0]['email']

print(f"\nAverage spam email length: {spam_emails.str.len().mean():.2f}")
print(f"Average ham email length: {ham_emails.str.len().mean():.2f}")

# Word count analysis
df['word_count'] = df['email'].apply(lambda x: len(str(x).split()))
print(f"\nAverage word count: {df['word_count'].mean():.2f} words")
print(f"Average spam word count: {spam_emails.apply(lambda x: len(str(x).split())).mean():.2f}")
print(f"Average ham word count: {ham_emails.apply(lambda x: len(str(x).split())).mean():.2f}")

# Most common words in spam vs ham
def get_top_words(emails, n=10):
    words = []
    for email in emails:
        # Simple tokenization: lowercase and split by non-alphanumeric
        tokens = re.findall(r'\b[a-zA-Z]+\b', str(email).lower())
        words.extend(tokens)
    return Counter(words).most_common(n)

print("\n=== Top 10 Words in Spam Emails ===")
spam_top_words = get_top_words(spam_emails, 10)
for word, count in spam_top_words:
    print(f"{word}: {count}")

print("\n=== Top 10 Words in Ham Emails ===")
ham_top_words = get_top_words(ham_emails, 10)
for word, count in ham_top_words:
    print(f"{word}: {count}")

# Save exploration results
exploration_results = {
    'dataset_shape': df.shape,
    'class_distribution': class_counts.to_dict(),
    'avg_email_length': float(df['email_length'].mean()),
    'avg_word_count': float(df['word_count'].mean()),
    'spam_avg_length': float(spam_emails.str.len().mean()),
    'ham_avg_length': float(ham_emails.str.len().mean()),
    'spam_top_words': spam_top_words,
    'ham_top_words': ham_top_words
}

print("\n=== Exploration Complete ===")
print("Key insights for ML modeling:")
print(f"- Dataset is balanced: {class_counts[0]} ham, {class_counts[1]} spam")
print(f"- Spam emails tend to be shorter ({spam_emails.str.len().mean():.1f} vs {ham_emails.str.len().mean():.1f} chars)")
print(f"- Common spam indicators: {', '.join([w[0] for w in spam_top_words[:5]])}")

# Save to file for reference
# Simple conversion for our results
def convert_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj

# Handle the results dictionary
def prepare_for_json(obj):
    if isinstance(obj, dict):
        return {key: prepare_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [prepare_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj

json_safe_results = prepare_for_json(exploration_results)

with open('ml_exploration_results.json', 'w') as f:
    json.dump(json_safe_results, f, indent=2)

print("Exploration results saved to ml_exploration_results.json")

# Show the results file
print("\n=== Saved Results Preview ===")
with open('ml_exploration_results.json', 'r') as f:
    print(f.read())