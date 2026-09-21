"""
ML-focused preprocessing and feature extraction for spam email detection
Pure ML side - no backend API or frontend integration
"""

import pandas as pd
import numpy as np
import re
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
def download_nltk_data():
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords')
    ]
    for path, name in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK data: {name}")
            nltk.download(name)

download_nltk_data()

def preprocess_text(text):
    """
    Preprocess a single email text:
    - Convert to lowercase
    - Remove punctuation and numbers
    - Tokenize
    - Remove stopwords
    - Apply stemming
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join back to string
    return ' '.join(tokens)

def main():
    print("Loading dataset...")
    df = pd.read_csv('spam_emails.csv')

    print(f"Dataset shape: {df.shape}")

    # Apply preprocessing
    print("Preprocessing email text...")
    df['cleaned_email'] = df['email'].apply(preprocess_text)

    # Show some examples
    print("\n=== Preprocessing Examples ===")
    for i in range(min(3, len(df))):
        print(f"Original: {df.iloc[i]['email']}")
        print(f"Cleaned:  {df.iloc[i]['cleaned_email']}")
        print()

    # Feature extraction using TF-IDF
    print("Extracting features using TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,      # Limit to top 5000 features
        ngram_range=(1, 2),     # Use unigrams and bigrams
        min_df=2,               # Ignore terms that appear in less than 2 documents
        max_df=0.95             # Ignore terms that appear in more than 95% of documents
    )

    # Fit and transform the cleaned emails
    X_features = tfidf_vectorizer.fit_transform(df['cleaned_email'])
    y_labels = df['label'].values

    print(f"Feature matrix shape: {X_features.shape}")
    print(f"Number of features: {X_features.shape[1]}")

    # Split the data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels,
        test_size=0.2,
        random_state=42,
        stratify=y_labels
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Save the processed data and vectorizer
    print("Saving preprocessed data and vectorizer...")

    # Save the vectorizer
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    # Save the processed data (sparse matrices need special handling)
    # We'll save them as numpy arrays after converting to dense (for small dataset)
    # Alternatively, we can save them in sparse format, but for simplicity we'll convert to dense
    # Note: For larger datasets, consider saving in sparse format using scipy.sparse.save_npz

    # Since our dataset is small, we can convert to dense
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    np.save('X_train.npy', X_train_dense)
    np.save('X_test.npy', X_test_dense)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    # Also save the cleaned dataframe for reference
    df.to_csv('processed_emails.csv', index=False)

    # Save feature names for interpretability
    feature_names = tfidf_vectorizer.get_feature_names_out()
    np.save('feature_names.npy', feature_names)

    print("\n=== Processing Complete ===")
    print("Saved files:")
    print("- tfidf_vectorizer.pkl: TF-IDF vectorizer")
    print("- X_train.npy, X_test.npy: Feature arrays")
    print("- y_train.npy, y_test.npy: Labels")
    print("- processed_emails.csv: Cleaned emails with labels")
    print("- feature_names.npy: Names of the TF-IDF features")

    # Show some feature names and their TF-IDF scores for the first spam email
    if X_train.shape[0] > 0:
        print("\n=== Top Features for First Training Email (Spam) ===")
        first_email_vector = X_train[0]  # This is a sparse matrix row (1 x n_features)
        # Get indices of non-zero elements
        nonzero_indices = first_email_vector.nonzero()[1]
        # Get the corresponding feature names and scores
        top_features = [(feature_names[i], first_email_vector[0, i]) for i in nonzero_indices]
        # Sort by score descending
        top_features.sort(key=lambda x: x[1], reverse=True)
        # Show top 10
        for feature, score in top_features[:10]:
            print(f"{feature}: {score:.4f}")

if __name__ == "__main__":
    main()