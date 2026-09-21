"""
ML training script - Email Spam Detection
=========================================

What this script does (matches the MODEL FLOW in the project architecture):
  1. Read the labelled datasets (team dataset + Enron-Spam if downloaded)
  2. Separate email text (X) and spam/ham label (y)
  3. Convert text into numerical features (TF-IDF)
  4. Split into train / test data
  5. Train three classifiers and evaluate them on emails they have NOT seen
  6. Pick the best one, retrain it on all data, and save ONE file: model.pkl

model.pkl is a scikit-learn Pipeline (TF-IDF + classifier), so exactly the
same text preprocessing is applied automatically at prediction time.

Usage:
    python download_dataset.py    # once, gets the large dataset
    python train_model.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE = Path(__file__).parent
TEAM_CSV = BASE / "spam_emails.csv"                 # 40 hand-made emails
ENRON_CSV = BASE / "data" / "enron_spam_data.csv"   # ~33,000 real emails
MODEL_PATH = BASE / "model.pkl"
METRICS_PATH = BASE / "metrics.json"

RANDOM_STATE = 42
MAX_CHARS = 3000   # only the first 3000 characters of each email are used
TEAM_WEIGHT = 5    # team emails are repeated so they count a bit more


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_team_data():
    df = pd.read_csv(TEAM_CSV)[["email", "label"]]
    return df.dropna().reset_index(drop=True)


def load_enron_data():
    df = pd.read_csv(ENRON_CSV)
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")
    df["email"] = (df["Subject"] + ". " + df["Message"].str[:MAX_CHARS]).str.strip()
    df["label"] = (df["Spam/Ham"] == "spam").astype(int)
    df = df[df["email"].str.len() > 10].drop_duplicates("email")
    return df[["email", "label"]].reset_index(drop=True)


# ----------------------------------------------------------------------
# Model definitions
# ----------------------------------------------------------------------
def build_pipeline(classifier, small_data=False):
    """TF-IDF text features + a classifier, bundled as one object."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),          # single words and word pairs
            sublinear_tf=True,
            min_df=1 if small_data else 3,
            max_df=1.0 if small_data else 0.9,
            max_features=100_000,
        )),
        ("clf", classifier),
    ])


def candidate_models(small_data=False):
    return {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=10, max_iter=3000, class_weight="balanced"),
        # LinearSVC has no probabilities, so it is wrapped to provide them
        "SVM": CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced"),
            cv=2 if small_data else 3),
    }


def score(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    team = load_team_data()
    print(f"Team dataset: {len(team)} emails")

    small_data = not ENRON_CSV.exists()
    results = {}

    if not small_data:
        # Recommended path: large dataset --------------------------------
        enron = load_enron_data()
        print(f"Enron-Spam dataset: {len(enron)} emails "
              f"({enron['label'].mean():.0%} spam)")

        train, test = train_test_split(
            enron, test_size=0.2, stratify=enron["label"],
            random_state=RANDOM_STATE)
        print(f"Train: {len(train)}   Test (never used for training): {len(test)}\n")

        for name, clf in candidate_models().items():
            model = build_pipeline(clf).fit(train["email"], train["label"])
            on_test = score(test["label"], model.predict(test["email"]))
            # The team's 40 emails were not part of this run, so they act as
            # a second, independent test set (different writing style).
            on_team = score(team["label"], model.predict(team["email"]))
            results[name] = {"enron_test": on_test, "team_emails": on_team}
            print(f"{name:20s} F1={on_test['f1']:.4f}  acc={on_test['accuracy']:.4f}"
                  f"  | team-email acc={on_team['accuracy']:.3f}")

        best_name = max(results, key=lambda n: (
            results[n]["enron_test"]["f1"], results[n]["team_emails"]["accuracy"]))

        # Final model learns from ALL data (more data = better model)
        final_data = pd.concat([enron] + [team] * TEAM_WEIGHT, ignore_index=True)
        note = ("Scores come from the held-out 20% of Enron-Spam and from the "
                "team's 40 emails, both unseen during that run. The saved model "
                "was then retrained on all data.")
    else:
        # Fallback path: only 40 emails ----------------------------------
        print("\nEnron dataset not found. Run `python download_dataset.py` for a "
              "much stronger model.\nFalling back to the 40-email team dataset "
              "(results are NOT reliable with so little data).\n")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for name, clf in candidate_models(small_data=True).items():
            pred = cross_val_predict(build_pipeline(clf, small_data=True),
                                     team["email"], team["label"], cv=cv)
            results[name] = {"team_emails_5fold_cv": score(team["label"], pred)}
            print(f"{name:20s} 5-fold CV accuracy="
                  f"{results[name]['team_emails_5fold_cv']['accuracy']:.3f}")
        best_name = max(results, key=lambda n: results[n]["team_emails_5fold_cv"]["f1"])
        final_data = team
        note = "5-fold cross-validation on only 40 emails - not reliable."

    print(f"\nBest model: {best_name}")

    final_model = build_pipeline(
        candidate_models(small_data)[best_name], small_data=small_data)
    final_model.fit(final_data["email"], final_data["label"])

    joblib.dump(final_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "best_model": best_name,
        "training_emails": int(len(final_data)),
        "note": note,
        "results": results,
    }, indent=2))
    print(f"Saved {MODEL_PATH.name} and {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
