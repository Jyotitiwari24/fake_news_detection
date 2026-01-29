from flask import Flask, render_template, request
import pickle
from utils import preprocess_text

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']           # Get news from form
    processed = preprocess_text(news)     # Preprocess the text
    vector = vectorizer.transform([processed])  # Convert to numerical features
    prediction = model.predict(vector)[0]       # Predict

    # Map prediction to human-readable result
    result = "Real News 🟢" if prediction == 0 else "Fake News 🔴"
    return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
