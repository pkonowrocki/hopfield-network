import dataManager
import dataCorruptor as dc
import HopfieldNetwork
import matplotlib.pyplot as plt
import numpy as np

def test0():
    X, size = dataManager.importData('data/large-25x25.csv')
    print("Loaded ", len(X), " samples")
    net = HopfieldNetwork.HopefieldNetwork(size)
    net.trainOja(X, u=0.001, iter=2000, showWeights=False)
    # net.trainHebb(X, showWeights=True)
    

    checkManuallyIfItWorks(net, X)

def checkManuallyIfItWorks(net, X):
    corruptedFigures2p = dc.corruptAllFigures(X, percent = 2, resultSizePerFigure = 1)
    corruptedFigures5p = dc.corruptAllFigures(X, percent = 5, resultSizePerFigure = 1)
    corruptedFigures10p = dc.corruptAllFigures(X, percent = 10, resultSizePerFigure = 1)
    corruptedFigures20p = dc.corruptAllFigures(X, percent = 20, resultSizePerFigure = 1)
    corruptedFigures30p = dc.corruptAllFigures(X, percent = 30, resultSizePerFigure = 1)
    corruptedFigures40p = dc.corruptAllFigures(X, percent = 40, resultSizePerFigure = 1)
    corruptedFigures50p = dc.corruptAllFigures(X, percent = 50, resultSizePerFigure = 1)

    plt.ioff()
    index = 0
    print('original')
    x = X[index]
    s = dataManager.resize(x)
    dataManager.show([s])

    print('2%')
    manuallyCheckForFigure(net, corruptedFigures2p[index][0])
    manuallyCheckForFigure(net, corruptedFigures2p[index+1][0])

    print('5%')
    manuallyCheckForFigure(net, corruptedFigures5p[index][0])
    manuallyCheckForFigure(net, corruptedFigures5p[index+1][0])

    print('10%')
    manuallyCheckForFigure(net, corruptedFigures10p[index][0])
    manuallyCheckForFigure(net, corruptedFigures10p[index+1][0])

    print('20%')
    manuallyCheckForFigure(net, corruptedFigures20p[index][0])
    manuallyCheckForFigure(net, corruptedFigures20p[index+1][0])

    print('30%')
    manuallyCheckForFigure(net, corruptedFigures30p[index][0])
    manuallyCheckForFigure(net, corruptedFigures30p[index+1][0])

    print('40%')
    manuallyCheckForFigure(net, corruptedFigures40p[index][0])
    manuallyCheckForFigure(net, corruptedFigures40p[index+1][0])

    print('50%')
    manuallyCheckForFigure(net, corruptedFigures50p[index][0])
    manuallyCheckForFigure(net, corruptedFigures50p[index+1][0])

def manuallyCheckForFigure(net, figure):
    s = dataManager.resize(figure)
    dataManager.show([s])
    s = net.forward(figure)
    s = dataManager.resize(s)
    dataManager.show([s])

def testAccuracyOfTrainingMethods(hebb = None, oja = None, corruptBy = 5):
    X, size = dataManager.importData('data/small-7x7.csv')

    if oja == None:
        oja = HopfieldNetwork.HopefieldNetwork(size)
        oja.trainOja(X, u=0.001, iter=2000, showWeights=False)

    if hebb == None:
        hebb = HopfieldNetwork.HopefieldNetwork(size)
        hebb.trainHebb(X, showWeights=False)

    Xc = dc.corruptAllFigures(X, percent = corruptBy, resultSizePerFigure = 5)

    correctOja = 0
    correctHebb = 0
    tries = 0;
    for i in range(len(Xc)):
        for j in range(len(Xc[i])):
            resultOja = oja.forward(data=Xc[i][j], iter=20)
            resultHebb = hebb.forward(data=Xc[i][j], iter=20)
            print(f'sample: {i+1}-{j+1}/{len(Xc)}-{len(Xc[i])} \tHebb: {np.all(resultHebb==X[i])}\tOja: {np.all(resultOja==X[i])}')
            tries += 1
            if np.all(resultHebb==X[i]):
                correctHebb += 1

            if np.all(resultOja==X[i]):
                correctOja +=1

            print(f'Oja learning rule got {correctOja}/{tries} right')
            print(f'Hebb learning rule got {correctHebb}/{tries} right')

if __name__ == "__main__":
    # test0()
    testAccuracyOfTrainingMethods()
