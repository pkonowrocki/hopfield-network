import dataManager
import dataCorruptor as dc
import HopfieldNetwork
import numpy as np

def test0():
    X, size = dataManager.importData('data/small-7x7.csv')
    net = HopfieldNetwork.HopefieldNetwork(size)
    net.trainOja(X, iter=100000)

    corruptedFigures = dc.corruptAllFigures(X, percent = 5, resultSizePerFigure = 10)
    index = 0
    
    x = X[index]
    s = dataManager.resize(x)
    dataManager.show([s])

    x = corruptedFigures[index][0]
    s = dataManager.resize(x)
    dataManager.show([s])
    s = net.forward(x)
    s = dataManager.resize(s)
    dataManager.show([s])

if __name__ == "__main__":
    test0()
