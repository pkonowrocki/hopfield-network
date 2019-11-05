import dataManager
import HopfieldNetwork

def test0():
    X, size = dataManager.importData('data/small-7x7.csv')
    net = HopfieldNetwork.HopefieldNetwork(size)
    net.trainHebb(X)
    dataManager.show(dataManager.resize(X))

if __name__ == "__main__":
    test0()
