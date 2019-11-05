import dataManager
import HopfieldNetwork

def test0():
    X, size = dataManager.importData('data/small-7x7.csv')
    net = HopfieldNetwork.HopefieldNetwork(size)
    net.trainHebb(X)
    y = X[1]
    dataManager.print(dataManager.resize(y))

if __name__ == "__main__":
    test0()
