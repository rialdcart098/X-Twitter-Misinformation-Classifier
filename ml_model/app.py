from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import dill as pickle
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

with open('model.pkl', 'rb') as fin:
    model = pickle.load(fin)
app = Flask(__name__)
CORS(app)
stop_words = set(stopwords.words('english') + list(string.punctuation))
def preprocess_text(posts: np.ndarray):
    """
    :param posts: np.ndarray - Input data to preprocess
    :return: np.ndarray or string - The preprocessed text data
    """
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    if isinstance(posts, str):
        posts = np.array([posts])
    processed_posts = []
    for i, post in enumerate(posts):
        post = np.array([i.lower() for i in word_tokenize(post) if i.lower() not in stop_words]).astype(posts.dtype)
        processed_posts.append(post)
    if len(posts) == 1:
        return processed_posts[0]
    return np.array(processed_posts, dtype=object)
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
    print(prediction[0])
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