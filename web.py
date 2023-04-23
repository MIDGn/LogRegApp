"""Libraries and metrics"""
import re
import time
import joblib
import spacy
import memory_profiler
from flask import Flask, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load pre-trained logistic regression model and vectorizer
model = joblib.load("model.pkl")
# Load CountVectorizer object
cv = joblib.load("cv.pkl")

# Load spaCy English language model for text preprocessing
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """ Preprocess input text """
    text = re.sub('[-,\\.!?]', '', text).lower()
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(lemmas)

def predict_text(text):
    """ Predict truthfulness of input text """
    text = preprocess_text(text)
    text_vect = cv.transform([text])
    pred = model.predict(text_vect)
    proba = model.predict_proba(text_vect)
    return int(pred), proba

@app.route("/", methods=["GET", "POST"])
def home():
    """ Home page with input form """
    if request.method == "POST":
        text = request.form["text"]
        start_time = time.time()
        mem_usage = memory_profiler.memory_usage()[0]
        pred, proba = predict_text(text)
        pred = 'True' if pred == 1 else 'Fake'
        prob_true = f"{proba[0][1]:.4f}"
        prob_fake = f"{proba[0][1]:.4f}"
        end_time = time.time()
        proc_time = round((end_time - start_time), 5)
        mem_usage_diff = round((memory_profiler.memory_usage()[0] - mem_usage), 5)
        return render_template("result.html", pred=pred, proba=[prob_fake, prob_true], text=text,
                                proc_time=proc_time, mem_usage=mem_usage_diff)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
