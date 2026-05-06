import numpy as np
import os
from network import NeuralNetwork
from preprocessing import load_data, preprocess, normalize, one_hot_encoding, decode
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

os.makedirs("model", exist_ok=True)
np.random.seed(41)


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

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prepare_data()

    input_label = X_train.shape[1]

    network = NeuralNetwork(
        architecture=[input_label, 256, 128, 10],
        n_iters=30,
        learning_rate=0.02,
        batch_size=16
    )

    network.fit(X_train, y_train)

    prediction = network.predict(X_test)

    y_test_decoded = decode(y_test)

    accuracy = accuracy_score(y_test_decoded, prediction)

    print(f"Final Accuracy: {accuracy}")
