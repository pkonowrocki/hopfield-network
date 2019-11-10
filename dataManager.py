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
    if type(X) is list:
        if size is None:
            size = _calculate_size(X[0].shape)
        return [_resize_single(x, size) for x in X]
    else:
        return _resize_single(X, size)

def _resize_single(X, size = None):
    if size is None:
        size = _calculate_size(X.shape)
    return np.resize(X, size)

def _calculate_size(shape):
    return (int(np.sqrt(shape)), int(np.sqrt(shape)))

def show(X, title = ''):
    plt.figure(1000)
    plt.ion()
    if type(X) is list:
        for i in range(len(X)):
            _show_single(X[i], f'{title} #{i+1}')
    else:
        _show_single(X, title)

def _show_single(X, title = ''):
    plt.figure(1000)
    plt.title(title)
    plt.imshow(X)
    plt.show()
    plt.pause(0.001)
