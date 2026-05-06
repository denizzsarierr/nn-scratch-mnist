import numpy as np
import os
import time
from network import NeuralNetwork
from preprocessing import load_data, preprocess, normalize, one_hot_encoding, decode
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

os.makedirs("model", exist_ok=True)
np.random.seed(41)



def prepare_data():

    df = load_data()

    X, y = preprocess(df)

    X = X.to_numpy()
    y = y.to_numpy().astype(int)

    X_normalized = normalize(X)
    y_encoded = one_hot_encoding(y,10)

    X_train, X_temporary, y_train, y_temporary = train_test_split(X_normalized,y_encoded,train_size=0.6,random_state=41)

    X_val, X_test, y_val, y_test = train_test_split(X_temporary,y_temporary,train_size=0.5,random_state=41)

    return X_train, X_val, X_test, y_train, y_val, y_test

def train(parameters, X_train, y_train, X_val, y_val):

    network = NeuralNetwork(architecture=parameters["architecture"],
                            n_iters = parameters["n_iters"],
                            learning_rate = parameters["learning_rate"],
                            batch_size= parameters["batch_size"])

    start = time.time()
    network.fit(X_train, y_train)
    end = time.time()

    prediction = network.predict(X_val)

    y_decoded = decode(y_val)

    accuracy = accuracy_score(y_decoded,prediction)

    return accuracy, end - start



    















                    




    











    