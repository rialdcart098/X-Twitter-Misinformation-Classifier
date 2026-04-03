from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import dill as pickle
import psycopg2
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


def get_db_connection():
    conn = psycopg2.connect(
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn

def preprocess_text(tweet_text: str):
    """
    :param tweet_text: str - Tweet text to preprocess
    :return: np.ndarray or string - The preprocessed text data
    """

    # CODE GRAVEYARD (RIP)
    # link_id = link.split('/')[-1]
    # BEARER_TOKEN = os.getenv("BEARER_TOKEN")
    # headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    # print(headers)
    # response = requests.get(f"https://api.twitter.com/2/tweets/{link_id}", headers=headers)
    # response.raise_for_status()
    # tweet_text = response.json()['data']['text']

    tweet_data = vectorizer.transform([tweet_text])
    print(type(tweet_data))
    return tweet_data, tweet_text
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to get predictions from the model
    Expects a JSON payload with a 'tweet' field
    :return: JSON - The prediction result
    """
    data = request.get_json()
    tweet = data['tweet']
    processed_tweet, tweet_text = preprocess_text(tweet)
    certainties = model.predict_proba(processed_tweet)
    prediction = np.argmax(certainties, axis=1)
    confidence = np.round(certainties[0, prediction[0]], 2)
    print(prediction)
    return jsonify({'prediction': bool(prediction[0]), 'confidence': float(confidence), 'tweet': tweet_text})
    # return jsonify({ 'prediction': True, 'confidence': 0.95, 'tweet': 'With his Golden Dome announcement today, @POTUS outlined a bold vision for layered defense to safeguard the homeland. We are ready now to support this mission with combat-proven systems and an open systems architecture that integrates the best of American technology.' })
@app.route('/feedback', methods=['POST'])
def feedback():
    """
    API endpoint to receive feedback on predictions
    Expects a JSON payload with 'tweet', 'prediction', and 'correct' fields
    :return: JSON - Acknowledgment of feedback receipt
    """
    data = request.get_json()
    tweet = data['tweet_text']
    prediction = bool(data['prediction'])

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tweets (tweet_text, majority_target) VALUES (%s, %s)",
        (tweet, prediction)
    )
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({'message': 'Feedback received successfully'})

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    :return: JSON - Health status
    """
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(port=1337)