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


def grid_search(X_train, y_train, X_val, y_val, input_label):


    parameters_grid = {"architecture" : [[input_label,64,10],
                                         [input_label,128,64,10],
                                         [input_label,256,128,10]],
                       "n_iters" : [20,30],
                       "learning_rate" : [0.02,0.01,0.005],
                       "batch_size" : [16]}

    best_score = 0
    best_parameters = None
    results = []

    for epoch in parameters_grid["n_iters"]:
        for lr in parameters_grid["learning_rate"]:
            for batch in parameters_grid["batch_size"]:
                for arch in parameters_grid["architecture"]:

                    parameters = {"architecture": arch,
                                  "n_iters": epoch,
                                  "learning_rate": lr,
                                  "batch_size": batch}


                    accuracy, runtime = train(parameters=parameters,
                                        X_train = X_train,
                                        y_train = y_train,
                                        X_val = X_val,
                                        y_val = y_val)

                    results.append((accuracy,runtime,parameters))

                    print(f"Parameters: {parameters}, Accuracy: {accuracy}, Runtime: {runtime:.2f}")

                    if accuracy > best_score:

                        best_score = accuracy
                        best_parameters = parameters

    return best_parameters, best_score, results
    

def parameter_tuning(X_train, y_train, X_val, y_val, input_label):

    print("------Tuning------")

    best_parameters, best_score, results = grid_search(X_train = X_train,
                                                       y_train = y_train,
                                                       X_val = X_val,
                                                       y_val = y_val,
                                                       input_label = input_label)


    print(f"Best Parameters: {best_parameters}")
    print(f"Best accuracy in validation set: {best_score}")

    return best_parameters, best_score, results

if __name__ == "__main__":

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()

    input_label = X_train.shape[1]

    best_parameters, best_score, results = parameter_tuning(X_train = X_train,
                                                            y_train = y_train,
                                                            X_val = X_val,
                                                            y_val = y_val,
                                                            input_label = input_label)

    
    print(f"Best Parameters: {best_parameters}, Best score: {best_score}")













                    




    











    