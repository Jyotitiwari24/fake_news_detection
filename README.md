# Fake News Detection Web App 📰

A Flask-based web app that detects whether a news article is **Real or Fake** using **TF-IDF** and **Logistic Regression**, fully **dockerized** for easy deployment.

---

## Features
- Real-time fake news detection from user input.
- Text preprocessing: stemming, stopword removal, TF-IDF vectorization.
- Web interface with Flask + HTML.
- Dockerized for easy deployment.

---

## Project Structure
fake-news-detector/
├── model/ # Saved model & vectorizer
├── templates/ # Frontend HTML
├── app.py # Flask app
├── utils.py # Text preprocessing
├── train_model.py # Train & save model
├── requirements.txt # Dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md


---

## Installation & Usage

### Clone repo
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
Run with Docker
docker-compose build
docker-compose up
Visit: http://localhost:5000

Or without Docker
pip install -r requirements.txt
python app.py
How it Works
User inputs news text.

Preprocessing: lowercase, remove non-letters, remove stopwords, stemming.

TF-IDF converts text to numeric features.

Logistic Regression predicts Real (0) or Fake (1).

Result displayed on web page.

Dependencies
Python 3.10+, Flask, scikit-learn, pandas, numpy, nltk, gunicorn, Docker