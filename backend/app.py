"""
Backend / prediction API - Email Spam Detection
===============================================

Loads the saved model ONCE at start-up (it is never retrained per request)
and exposes it to the frontend.

Endpoints
---------
POST /predict   body: {"email": "<text>"}
                reply: {"prediction": "spam" | "ham",
                        "label": "Spam" | "Not Spam",
                        "confidence": 0.0 - 1.0,
                        "spam_probability": 0.0 - 1.0}
GET  /health    quick check that the server and model are running
GET  /          serves the website (../frontend/index.html)

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import json
from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.pkl"
FRONTEND_DIR = (BASE.parent / "frontend").resolve()

MIN_CHARS = 5
MAX_CHARS = 20_000

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

if not MODEL_PATH.exists():
    raise SystemExit("model.pkl not found. Run `python train_model.py` first.")

# Pipeline = TF-IDF + classifier, so the same preprocessing as training is used.
model = joblib.load(MODEL_PATH)

# Name of the winning algorithm, written by train_model.py
try:
    MODEL_NAME = json.loads((BASE / "metrics.json").read_text())["best_model"]
except (OSError, KeyError, ValueError):
    MODEL_NAME = type(model.named_steps["clf"]).__name__


# ----------------------------------------------------------------------
# CORS - lets the page work even if index.html is opened from a different
# origin (e.g. VS Code Live Server on port 5500 or straight from disk).
# ----------------------------------------------------------------------
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


def error(message, status):
    return jsonify({"error": message}), status


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":          # browser pre-flight request
        return "", 204

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "email" not in data:
        return error('Send JSON like {"email": "your message"}.', 400)

    email = data["email"]
    if not isinstance(email, str) or not email.strip():
        return error("Email text must be a non-empty string.", 400)

    email = email.strip()
    if len(email) < MIN_CHARS:
        return error(f"Email text must be at least {MIN_CHARS} characters.", 400)
    if len(email) > MAX_CHARS:
        return error(f"Email text must be at most {MAX_CHARS} characters.", 400)

    # Classes are 0 = ham, 1 = spam
    spam_probability = float(model.predict_proba([email])[0][1])
    is_spam = spam_probability >= 0.5
    confidence = spam_probability if is_spam else 1.0 - spam_probability

    return jsonify({
        "prediction": "spam" if is_spam else "ham",
        "label": "Spam" if is_spam else "Not Spam",
        "confidence": round(confidence, 4),
        "spam_probability": round(spam_probability, 4),
    })


if __name__ == "__main__":
    print("Spam Detection API running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
