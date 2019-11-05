import csv
import numpy as np
import matplotlib.pyplot as plt

def importData(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        for row in csv_reader:
            X.append(np.array(row, dtype=float))    
    return X, np.max(X[1].shape)

def resize(X, size = None):
    if size is None:
        s = X[1].shape[1]
        size = (np.sqrt(s), np.sqrt(s))

    if len(X) > 1:
        return [np.resize(x, size) for x in X]
    else:
        return np.resize(X, size)

def print(X):
    plt.imshow(X)
    plt.show()