import numpy as np
from network import NeuralNetwork
from preprocessing import load_data, preprocess, normalize, one_hot_encoding, decode
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def prepare_data():

    df = load_data()

    X, y = preprocess(df)

    X = X.to_numpy()
    y = y.to_numpy().astype(int)

    X_normalized = normalize(X)
    y_encoded = one_hot_encoding(y, 10)

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_encoded, train_size=0.8, random_state=41
    )

    return X_test, y_test
