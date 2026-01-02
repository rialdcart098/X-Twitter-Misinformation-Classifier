from flask import Flask, request, jsonify
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

with open('model.pkl', 'rb') as fin:
    model = pickle.load(fin)
app = Flask(__name__)
stop_words = set(stopwords.words('english') + list(string.punctuation))
def preprocess_text(post: str) -> np.ndarray:
    """
    :param post: str - Input text to preprocess
    :return: np.ndarray - The preprocessed text data
    """
    processed_post = np.array([i.lower() for i in word_tokenize(post) if i.lower() not in stop_words])
    return processed_post
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
    prediction = model.predict([processed_tweet])
    return jsonify({'prediction': bool(prediction[0])})

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    :return: JSON - Health status
    """
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(port=1337, debug=True)