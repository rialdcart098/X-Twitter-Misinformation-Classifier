import string
from typing import Tuple
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import dill as pickle
import time

def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param data: pd.DataFrame - Input data to preprocess
    :return: tuple - (posts, labels) the preprocessed data
    """
    data = data.copy()
    clean_data = data.dropna()
    posts = clean_data['tweet'].values
    labels = clean_data['majority_target'].values.astype(int)
    return posts, labels

def get_time(func):
    """
    :param func: function - The function to time
    :return: function - The wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f}s")
        return result
    return wrapper


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

def split_data(x: np.ndarray, y: np.ndarray, train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :param x: np.ndarray - Input data to split
    :param y: np.ndarray - Output data to split
    :param train_size: float - Proportion of data to use for training
    :return: tuple - (posts_train, posts_test, labels_train, labels_test) the split datasets
    """
    return train_test_split(x, y, test_size=train_size, random_state=42, stratify=y)

# Metrics
def accuracy(tp: float, tn: float, fp: float, fn: float) -> float:
    if (tp + tn + fp + fn) == 0: return 0.0
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp: float, fp: float) -> float:
    if (tp + fp) == 0: return 0.0
    return tp / (tp + fp)

def recall(tp: float, fn: float) -> float:
    if (tp + fn) == 0: return 0.0
    return tp / (tp + fn)

def f1_score(tp: float, fp: float, fn: float) -> float:
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    if (prec + rec) == 0: return 0.0
    return 2 * (prec * rec) / (prec + rec)

def metrics(y: np.ndarray, y_hat: np.ndarray) -> None:
    """
    :param y: np.ndarray - True labels
    :param y_hat: np.ndarray - Predicted labels
    :return: None - Prints the accuracy, precision, recall, and F1-score of the predictions
    """
    true_positives = np.sum((y == True) & (y_hat == True))
    true_negatives = np.sum((y == False) & (y_hat == False))
    false_positives = np.sum((y == False) & (y_hat == True))
    false_negatives = np.sum((y == True) & (y_hat == False))
    print('-' * 5 + ' Metrics ' + '-' * 5)
    print(f'True positives: {true_positives}')
    print(f'True negatives: {true_negatives}')
    print(f'False positives: {false_positives}')
    print(f'False negatives: {false_negatives}')
    print(f'Accuracy: {np.round(accuracy(true_positives, true_negatives, false_positives, false_negatives), 2)}')
    print(f'Precision: {np.round(precision(true_positives, false_positives), 2)}')
    print(f'Recall: {np.round(recall(true_positives, false_negatives), 2)}')
    print(f'F1-Score: {np.round(f1_score(true_positives, false_positives, false_negatives), 2)}')
    print('-' * 20)

def main():
    df = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')
    labels = df['majority_target'].astype(int)
    vectorizer = TfidfVectorizer()
    posts = vectorizer.fit_transform(df['tweet'])  # keep as sparse
    posts_train, posts_test, labels_train, labels_test = train_test_split(
        posts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(posts_train, labels_train)

    # This block is for evaluating the model
    predictions = model.predict(posts_test)
    metrics(labels_test.values, predictions)

    # This block is for saving the model
    # with open('model.pkl', 'wb') as fin:
    #     pickle.dump(model, fin)
    # print('Model saved to model.pkl')
    # with open('vectorizer.pkl', 'wb') as fin:
    #     pickle.dump(vectorizer, fin)
    # print('Vectorizer saved to vectorizer.pkl')

if __name__ == "__main__":
    main()

