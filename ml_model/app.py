from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import dill as pickle
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import os
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
load_dotenv()

with open('model.pkl', 'rb') as fin:
    model = pickle.load(fin)
with open('vectorizer.pkl', 'rb') as fin:
    vectorizer = pickle.load(fin)
app = Flask(__name__)
CORS(app)
stop_words = set(stopwords.words('english') + list(string.punctuation))



def preprocess_text(link: str):
    """
    :param posts: np.ndarray - Input data to preprocess
    :return: np.ndarray or string - The preprocessed text data
    """
    linkID = link.split('/')[-1]
    BEARER_TOKEN = os.getenv("BEARER_TOKEN")
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    response = requests.get(f"https://api.twitter.com/2/tweets/{linkID}", headers=headers)
    response.raise_for_status()
    tweet_data = response.json()['data']['text']
    tweet_data = vectorizer.transform([tweet_data])
    print(type(tweet_data))
    return tweet_data
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to get predictions from the model
    Expects a JSON payload with a 'tweet' field
    :return: JSON - The prediction result
    """
    data = request.get_json()
    tweet = data['tweet']
    processed_tweet = preprocess_text(tweet)
    certainties = model.predict_proba(processed_tweet)
    prediction = np.argmax(certainties, axis=1)
    confidence = np.round(certainties[0, prediction[0]] * 100, 2)
    print(prediction)
    return jsonify({'prediction': bool(prediction[0]), 'confidence': float(confidence)})

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    :return: JSON - Health status
    """
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(port=1337, debug=True)
# curl -X GET https://api.twitter.com/2/tweets/1924971025655017833 -H "Authorization: Bearer AAAAAAAAAAAAAAAAAAAAAONz6gEAAAAAM3q2Ke8pMuAYpFjJaqJeL9iNIOA%3DNLfDJDCJFZbodXEhm31GZTMtHZRAfPjBLoWVCM407Ktfjawk4p