"""
ML-focused model training and evaluation for spam email detection
Pure ML side - no backend API or frontend integration
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_data():
    """Load the preprocessed data"""
    print("Loading preprocessed data...")

    # Load the feature arrays and labels
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    # Load feature names for interpretability
    feature_names = np.load('feature_names.npy', allow_pickle=True)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, feature_names

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models and evaluate their performance"""

    # Define models to train
    models = {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='linear', random_state=42, probability=True)
    }

    results = {}

    print("\n=== Model Training and Evaluation ===")

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        precision = precision_score(y_test, y_pred_test)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        # Store results
        results[name] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred_test': y_pred_test
        }

        print(f"{name} Results:")
        print(f"  Training Accuracy: {train_accuracy:.4f}")
        print(f"  Test Accuracy:     {test_accuracy:.4f}")
        print(f"  Precision:         {precision:.4f}")
        print(f"  Recall:            {recall:.4f}")
        print(f"  F1-Score:          {f1:.4f}")

    return results

def evaluate_best_model(results, X_test, y_test, feature_names):
    """Detailed evaluation of the best performing model"""

    # Find the best model based on F1-score
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']

    print(f"\n=== Best Model: {best_model_name} ===")

    # Detailed classification report
    y_pred_best = results[best_model_name]['y_pred_test']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion Matrix:")
    print(cm)
    print("(Rows: Actual, Columns: Predicted)")
    print("[ TN  FP ]")
    print("[ FN  TP ]")

    # Feature importance (for models that support it)
    if hasattr(best_model, 'coef_'):
        print(f"\nTop 10 Most Important Features for Spam Detection:")
        # For binary classification, we look at coefficients for the spam class (class 1)
        coefs = best_model.coef_[0]
        feature_importance = list(zip(feature_names, coefs))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print("Top spam indicators (positive coefficients):")
        for feature, coef in feature_importance[:10]:
            print(f"  {feature}: {coef:.4f}")

        print("\nTop ham indicators (negative coefficients):")
        for feature, coef in feature_importance[-10:]:
            print(f"  {feature}: {coef:.4f}")

    return best_model_name, best_model

def save_model_and_results(best_model_name, best_model, results):
    """Save the best model and evaluation results"""

    # Save the best model
    model_filename = f'best_spam_model_{best_model_name.lower().replace(" ", "_")}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest model saved as: {model_filename}")

    # Save evaluation results
    eval_results = {}
    for name, result in results.items():
        eval_results[name] = {
            'train_accuracy': float(result['train_accuracy']),
            'test_accuracy': float(result['test_accuracy']),
            'precision': float(result['precision']),
            'recall': float(result['recall']),
            'f1': float(result['f1'])
        }

    with open('model_evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    print("Evaluation results saved to: model_evaluation_results.json")

    # Also save all models for comparison
    with open('all_models.pkl', 'wb') as f:
        model_dict = {name: result['model'] for name, result in results.items()}
        pickle.dump(model_dict, f)
    print("All models saved to: all_models.pkl")

def main():
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # Train and evaluate models
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Evaluate best model in detail
    best_model_name, best_model = evaluate_best_model(results, X_test, y_test, feature_names)

    # Save model and results
    save_model_and_results(best_model_name, best_model, results)

    print("\n=== Training Complete ===")
    print("Summary of all models:")
    for name, result in results.items():
        print(f"{name:20} - Test Acc: {result['test_accuracy']:.4f}, F1: {result['f1']:.4f}")

if __name__ == "__main__":
    main()