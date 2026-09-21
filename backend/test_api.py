"""
Automated checks for the backend API.

Run (from the backend folder):
    python test_api.py
"""

from app import app

client = app.test_client()
passed = 0


def check(name, condition):
    global passed
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise SystemExit(1)
    passed += 1


def post(payload=None, **kwargs):
    return client.post("/predict", json=payload, **kwargs)


# --- Health -------------------------------------------------------------
r = client.get("/health")
check("GET /health returns ok", r.status_code == 200 and r.get_json()["status"] == "ok")

# --- Frontend is served -------------------------------------------------
r = client.get("/")
check("GET / serves the website", r.status_code == 200 and b"SpamGuard" in r.data)
check("style.css is served", client.get("/style.css").status_code == 200)
check("script.js is served", client.get("/script.js").status_code == 200)

# --- Predictions --------------------------------------------------------
spam_examples = [
    "CONGRATULATIONS!!! You are the lucky winner of a $5,000 gift card. Claim your reward now before it expires!",
    "Cheap medications online, no prescription needed. Order now and get 70% off.",
    "Win a free iPhone now! Click this link to claim your prize.",
]
ham_examples = [
    "Hi Sarah, could you send me the slides from yesterday's client presentation before noon?",
    "Reminder: the quarterly review meeting is moved to Thursday at 2 PM in room 4B.",
    "Please find attached the minutes of the board meeting. Kindly confirm receipt.",
]

for text in spam_examples:
    body = post({"email": text}).get_json()
    check(f"spam detected: {text[:40]}...", body["prediction"] == "spam")

for text in ham_examples:
    body = post({"email": text}).get_json()
    check(f"ham detected:  {text[:40]}...", body["prediction"] == "ham")

body = post({"email": spam_examples[0]}).get_json()
check("response has all fields",
      set(body) == {"prediction", "label", "confidence", "spam_probability"})
check("confidence is between 0.5 and 1", 0.5 <= body["confidence"] <= 1.0)

# --- Input validation ---------------------------------------------------
check("missing 'email' key -> 400", post({"text": "hello there"}).status_code == 400)
check("empty email -> 400", post({"email": "   "}).status_code == 400)
check("too short -> 400", post({"email": "hi"}).status_code == 400)
check("too long -> 400", post({"email": "a" * 20_001}).status_code == 400)
check("number instead of text -> 400", post({"email": 12345}).status_code == 400)
check("invalid JSON -> 400", client.post(
    "/predict", data="not json", content_type="application/json").status_code == 400)
check("error reply contains a message", "error" in post({"email": ""}).get_json())

# --- CORS ---------------------------------------------------------------
r = client.options("/predict")
check("OPTIONS pre-flight works", r.status_code == 204)
check("CORS header present", r.headers.get("Access-Control-Allow-Origin") == "*")

print(f"\nAll {passed} checks passed.")
