"""
ML-focused inference script for spam email detection
Pure ML side - no backend API or frontend integration
Demonstrates how to use the trained model to predict spam/ham for new emails
"""

import pickle
import numpy as np
import re
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
    Preprocess a single email text (same as used in training):
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

def load_model_and_vectorizer(model_path='best_spam_model_naive_bayes.pkl',
                             vectorizer_path='tfidf_vectorizer.pkl'):
    """Load the trained model and TF-IDF vectorizer"""
    print("Loading trained model and vectorizer...")

    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load the vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    print(f"Model loaded: {type(model).__name__}")
    print(f"Vectorizer loaded: {type(vectorizer).__name__}")

    return model, vectorizer

def predict_spam(email_text, model, vectorizer):
    """
    Predict whether an email is spam or not

    Returns:
        dict: Prediction results with probabilities and classification
    """
    # Preprocess the email
    cleaned_email = preprocess_text(email_text)

    # Vectorize the cleaned email
    email_vector = vectorizer.transform([cleaned_email])

    # Make prediction
    prediction = model.predict(email_vector)[0]  # 0 = ham, 1 = spam

    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(email_vector)[0]
        spam_probability = probabilities[1]  # Probability of being spam
        ham_probability = probabilities[0]   # Probability of being ham
    else:
        # For models without predict_proba (like some SVMs without probability=True)
        # We can use decision function or just return the prediction
        spam_probability = float(prediction)  # This is just 0.0 or 1.0
        ham_probability = 1.0 - spam_probability

    # Prepare result
    result = {
        'email': email_text,
        'cleaned_email': cleaned_email,
        'prediction': int(prediction),
        'predicted_label': 'Spam' if prediction == 1 else 'Ham',
        'spam_probability': float(spam_probability),
        'ham_probability': float(ham_probability),
        'is_spam': bool(prediction == 1)
    }

    return result

def main():
    print("=== ML Inference for Spam Email Detection ===\n")

    # Load the trained model and vectorizer
    try:
        model, vectorizer = load_model_and_vectorizer()
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please make sure you have run ml_train.py first to generate the model files.")
        return

    # Test with some example emails
    test_emails = [
        # Spam examples
        "WINNER! You have been selected to receive a free prize! Click here now!",
        "Urgent: Your account needs verification. Click link to avoid suspension.",
        "Congratulations! You've won $1,000,000. Call now to claim your prize!",
        "Free Viagra! Buy now and save 50%. Limited time offer!",

        # Legitimate/Ham examples
        "Hi John, can you please review the attached document before tomorrow's meeting?",
        "Thanks for your help with the project. Let's catch up for coffee next week?",
        "Please find the meeting minutes from yesterday's discussion attached.",
        "Your package has been shipped and will arrive on Friday. Tracking number: 12345"
    ]

    print("Making predictions on test emails:\n")

    for i, email in enumerate(test_emails, 1):
        print(f"--- Test Email {i} ---")
        result = predict_spam(email, model, vectorizer)

        print(f"Original: {result['email']}")
        print(f"Cleaned:  {result['cleaned_email']}")
        print(f"Prediction: {result['predicted_label']} "
              f"(Spam prob: {result['spam_probability']:.3f}, "
              f"Ham prob: {result['ham_probability']:.3f})")
        print(f"Result: {'SPAM' if result['is_spam'] else 'LEGITIMATE'}")
        print()

    # Interactive mode
    print("--- Interactive Mode ---")
    print("Enter email text to classify (type 'quit' to exit):")

    while True:
        user_input = input("\nEmail text: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            result = predict_spam(user_input, model, vectorizer)
            print(f"Prediction: {result['predicted_label']}")
            print(f"Confidence: {max(result['spam_probability'], result['ham_probability']):.3f}")
            print(f"Classification: {'SPAM' if result['is_spam'] else 'LEGITIMATE'}")
        except Exception as e:
            print(f"Error processing email: {e}")

if __name__ == "__main__":
    main()