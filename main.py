import string
from typing import Tuple
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import train_test_split

def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param data: pd.DataFrame - Input data to preprocess
    :return: tuple - (posts, labels) the preprocessed data
    """
    data = data.copy()
    clean_data = data.dropna()
    posts = clean_data['Text'].values
    labels = clean_data['label'].values
    return posts, labels

def split_data(x: np.ndarray, y: np.ndarray, train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :param x: np.ndarray - Input data to split
    :param y: np.ndarray - Output data to split
    :param train_size: float - Proportion of data to use for training
    :return: tuple - (posts_train, posts_test, labels_train, labels_test) the split datasets
    """
    return train_test_split(x, y, test_size=train_size, random_state=42, stratify=y)

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
    return np.array(processed_posts)


def main():
    df = pd.read_csv('fake_and_real_news.csv')
    posts, labels = preprocess_data(df)
    posts_train, posts_test, labels_train, labels_test = split_data(posts, labels)

    class NaiveBayes:
        def __init__(self) -> None:
            self.label_count = {}
            self.word_count = {}
        def fit(self, x: np.ndarray, y: np.ndarray) -> None:
            for i in range(len(x)):
                label = y[i]
                post = x[i]
                for word in post:
                    if word not in self.word_count:
                        self.word_count[word] = {'Fake': 0, 'Real': 0}
                    if label == 'Fake':
                        self.word_count[word]['Fake'] += 1
                    else:
                        self.word_count[word]['Real'] += 1
            self.label_count = {'Fake': sum(label.lower() == 'fake' for label in y), 'Real': sum(label.lower() == 'real' for label in y)}


if __name__ == "__main__":
    main()