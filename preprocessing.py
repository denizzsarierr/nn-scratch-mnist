import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

def load_data():

    mnist = fetch_openml("mnist_784")

    mnist_df = pd.DataFrame(data=mnist['data'],columns=mnist['feature_names'])
    mnist_df['target'] = mnist['target']

    return mnist_df

def preprocess(df):

    df = df.apply(pd.to_numeric)
    
    X = df.drop('target',axis = 1)
    y = df.target 

    return X,y


def normalize(X):

    X_normalized = X / 255.0

    return X_normalized

def one_hot_encoding(y,number_class):
    y_encode = np.zeros((y.size,number_class))
    indices = np.arange(y.size)
    y_encode[indices,y] = 1

    return y_encode


def decode(y):

    y_true = np.argmax(y,axis = 1)

    return y_true