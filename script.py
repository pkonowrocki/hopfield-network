import dataManager
import dataCorruptor as dc
import HopfieldNetwork
import matplotlib.pyplot as plt
import numpy as np

def test0():
    X, size = dataManager.importData('data/large-25x25.csv')
    print("Loaded ", len(X), " samples")
    net = HopfieldNetwork.HopefieldNetwork(size)
    net.trainOja(X, u=0.001, iter=200, showWeights=True)
    # net.trainHebb(X, showWeights=True)
    
    corruptedFigures = dc.corruptAllFigures(X, percent = 2, resultSizePerFigure = 10)
    
    plt.ioff()
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
