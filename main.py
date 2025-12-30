from typing import Tuple
import numpy as np
import pandas as pd
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
    :param data: pd.DataFrame - Input data to split
    :param train_size: float - Proportion of data to use for training
    :return: tuple - (posts_train, posts_test, labels_train, labels_test) the split datasets
    """
    return train_test_split(x, y, test_size=train_size, random_state=42, stratify=y)



def main():
    df = pd.read_csv('fake_and_real_news.csv')
    posts, labels = preprocess_data(df)
    posts_train, posts_test, labels_train, labels_test = split_data(posts, labels)


if __name__ == "__main__":
    main()