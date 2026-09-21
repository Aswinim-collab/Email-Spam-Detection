"""
Download the Enron-Spam dataset (~33,000 real, labelled emails).

Run once, before train_model.py:
    python download_dataset.py

Source: https://github.com/MWiechmann/enron_spam_data
Original paper: Metsis, Androutsopoulos & Paliouras (2006),
"Spam Filtering with Naive Bayes - Which Naive Bayes?"
"""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://raw.githubusercontent.com/MWiechmann/"
       "enron_spam_data/master/enron_spam_data.zip")
DATA_DIR = Path(__file__).parent / "data"
TARGET = DATA_DIR / "enron_spam_data.csv"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    if TARGET.exists():
        print(f"Already downloaded: {TARGET}")
        return
    print("Downloading Enron-Spam dataset (about 15 MB)...")
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extract("enron_spam_data.csv", DATA_DIR)
    print(f"Saved to {TARGET}")


if __name__ == "__main__":
    main()
