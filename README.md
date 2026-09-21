# 📧 Email Spam Detection Using AI and Machine Learning

## 📌 Project Overview
Email Spam Detection is an AI and Machine Learning-based web application that automatically classifies emails as **Spam** or **Not Spam**. The system analyzes the email content using Natural Language Processing and a trained Machine Learning model.

The project consists of three main components: **Machine Learning, Flask Backend, and Web Frontend**.

## 🎯 Objectives

- Detect spam emails automatically.
- Classify emails as Spam or Not Spam.
- Use Machine Learning for text classification.
- Provide a simple and user-friendly web interface.
- Connect the ML model with the frontend using Flask.

## 🛠️ Technologies Used

- **Python**
- **Pandas & NumPy**
- **Scikit-learn**
- **NLTK**
- **TF-IDF**
- **Support Vector Machine (SVM)**
- **Flask**
- **HTML**
- **CSS**
- **JavaScript**

## ⚙️ How It Works

```
User enters Email
	↓
Frontend
	↓
Flask Backend
	↓
Text Preprocessing
	↓
TF-IDF Feature Extraction
	↓
SVM Machine Learning Model
	↓
Spam / Not Spam
	↓
Result displayed to User
```

## 🔑 Features

- 📩 Email text input
- 🤖 Machine Learning-based prediction
- 🔍 Spam and Not Spam classification
- 📊 Prediction confidence
- ⚡ Fast response
- 💻 Simple web interface
- 🔗 Frontend and backend API integration

## 🧠 Machine Learning
The email text is preprocessed and converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

An **SVM (Support Vector Machine)** classifier is then used to predict whether the email is Spam or Not Spam.

The model is trained and evaluated using labelled email data.

## 🌐 Backend
The backend is developed using **Flask**.

It receives email content from the frontend, sends it to the trained Machine Learning model, and returns the prediction result.

## 🎨 Frontend
The frontend is developed using **HTML, CSS, and JavaScript**.

Users can enter or paste an email message and click the **Check Email** button to receive the prediction.

## 🚀 How to Run the Project

### 1. Clone the Repository

```
git clone <your-github-repository-link>
cd email-spam-detection
```

### 2. Install Required Libraries

```
pip install -r requirements.txt
```

### 3. Run the Flask Application

```
python app.py
```

### 4. Open in Browser
Open the local URL shown by Flask, usually:

```
http://127.0.0.1:5000/
```

## 📂 Project Structure

```
Email-Spam-Detection/
│
├── app.py
├── model/
│   └── trained_model
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── dataset/
│   └── email_dataset
│
├── requirements.txt
└── README.md
```

## 📈 Expected Output

### Spam Email

```
Input:
Congratulations! You have won a free prize. Click here to claim.

Output:
SPAM DETECTED
```

### Normal Email

```
Input:
Please attend the meeting tomorrow at 10 AM.

Output:
NOT SPAM
```

## 🔮 Future Enhancements

- Phishing URL detection
- Malicious attachment detection
- Direct email account integration
- Larger and updated datasets
- Deep Learning-based classification
- Multilingual spam detection
- Cloud deployment

## 👥 Project Contribution
The project was developed by dividing the work into three components:

- **Machine Learning** – Dataset processing, model training and evaluation
- **Backend** – Flask API and model integration
- **Frontend** – Web interface and user interaction

## 📄 Conclusion
This project demonstrates the practical use of **Artificial Intelligence, Machine Learning, Natural Language Processing, and Web Development** to detect spam emails. It provides a simple interface for users to check emails and receive an automated Spam or Not Spam prediction.
