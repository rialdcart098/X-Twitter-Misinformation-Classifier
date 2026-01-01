import string
from typing import Tuple
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
import time

def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param data: pd.DataFrame - Input data to preprocess
    :return: tuple - (posts, labels) the preprocessed data
    """
    data = data.copy()
    clean_data = data.dropna()
    posts = clean_data['tweet'].values
    labels = clean_data['majority_target'].values
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

def accuracy(tp: float, tn: float, fp: float, fn: float) -> float:
    """
    :param tp: float - True Positives
    :param tn: float - True Negatives
    :param fp: float - False Positives
    :param fn: float - False Negatives
    :return: float - The accuracy of the predictions
    """
    if (tp + tn + fp + fn) == 0: return 0.0
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp: float, fp: float) -> float:
    """
    :param tp: float - True Positives
    :param fp: float - False Positives
    :return: float - The precision of the predictions
    """
    if (tp + fp) == 0: return 0.0
    return tp / (tp + fp)

def recall(tp: float, fn: float) -> float:
    """
    :param tp: float - True Positives
    :param fn: float - False Negatives
    :return: float - The recall of the predictions
    """
    if (tp + fn) == 0: return 0.0
    return tp / (tp + fn)

def f1_score(tp: float, fp: float, fn: float) -> float:
    """
    :param tp: float - True Positives
    :param fp: float - False Positives
    :param fn: float - False Negatives
    :return: float - The F1-score of the predictions
    """
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
    true_positives = np.sum((y == False) & (y_hat == False))
    true_negatives = np.sum((y == True) & (y_hat == True))
    false_positives = np.sum((y == True) & (y_hat == False))
    false_negatives = np.sum((y == False) & (y_hat == True))
    print('-' * 5 + ' Metrics ' + '-' * 5)
    print(f'Accuracy: {np.round(accuracy(true_positives, true_negatives, false_positives, false_negatives), 2)}')
    print(f'Precision: {np.round(precision(true_positives, false_positives), 2)}')
    print(f'Recall: {np.round(recall(true_positives, false_negatives), 2)}')
    print(f'F1-Score: {np.round(f1_score(true_positives, false_positives, false_negatives), 2)}')
    print('-' * 20)
def main():
    df = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')
    posts, labels = preprocess_data(df)
    posts_train, posts_test, labels_train, labels_test = split_data(posts, labels)

    class NaiveBayes:
        def __init__(self) -> None:
            self.label_count = {False: 0, True: 0}
            self.word_count = {}
            self.total_words_in_class = {False: 0, True: 0}
        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            for i in range(len(x)):
                label = y[i]
                post = x[i]
                self.label_count[label] += 1
                for word in post:
                    if word not in self.word_count:
                        self.word_count[word] = {False: 0, True: 0}
                    self.word_count[word][label] += 1
                    self.total_words_in_class[label] += 1
        def laplace_smoothing(self, word: str, label: bool) -> float:
            if word in self.word_count:
                count = self.word_count[word][label]
            else:
                count = 0
            p_word_given_label = (count + 1) / (self.total_words_in_class[label] + len(self.word_count))
            return p_word_given_label
        def calculate_prior(self, label: bool) -> float:
            return self.label_count[label] / (self.label_count[False] + self.label_count[True])
        @get_time
        def predict(self, x: np.ndarray) -> np.ndarray:
            guesses = []
            for i in range(len(x)):
                post = x[i]
                p_fake = self.calculate_prior(False)
                p_real = self.calculate_prior(True)
                for word in post:
                    p_fake += np.log(self.laplace_smoothing(word, False))
                    p_real += np.log(self.laplace_smoothing(word, True))
                if p_fake > p_real:
                    guesses.append(False)
                else:
                    guesses.append(True)
            return np.array(guesses, dtype=object)
    model = NaiveBayes()
    model.fit(preprocess_text(posts_train), labels_train)
    predictions = model.predict(preprocess_text(posts_test))
    metrics(labels_test, predictions)

if __name__ == "__main__":
    main()