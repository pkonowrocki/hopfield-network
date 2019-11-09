import dataManager
import HopfieldNetwork
import numpy as np

def test0():
    X, size = dataManager.importData('data/small-7x7.csv')
    net = HopfieldNetwork.HopefieldNetwork(size)
    net.trainOja(X)
    x = X[4]
    s = dataManager.resize(x)
    dataManager.show([s])
    x[5] = x[5]*(-1)
    x[20] = x[20]*(-1)
    x[15] = x[15]*(-1)
    s = dataManager.resize(x)
    dataManager.show([s])
    s = net.forward(x)
    s = dataManager.resize(s)
    dataManager.show([s])

if __name__ == "__main__":
    test0()
