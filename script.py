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

def testAccuracyOfTrainingMethods(hebb = None, oja = None, corruptBy = 0, showImages = True):
    X, size = dataManager.importData('data/small-7x7.csv')

    if oja == None:
        oja = HopfieldNetwork.HopefieldNetwork(size)
        oja.trainOja(X, u=0.001, iter=2000, showWeights=False)

    if hebb == None:
        hebb = HopfieldNetwork.HopefieldNetwork(size)
        hebb.trainHebb(X, showWeights=False)

    Xc = dc.corruptAllFigures(X, percent = corruptBy, resultSizePerFigure = 1)

    correctOja = 0
    correctHebb = 0
    for i in range(len(X)):
        if showImages:
            dataManager.show([dataManager.resize(Xc[i][0])], f'orginal')
            
            dataManager.stopAnimation()

            resultHebb = hebb.forward(
                data=Xc[i][0], 
                iter=20000, 
                asyn=False, 
                print=lambda c, t: dataManager.show([dataManager.resize(c)], f'Hebb iteration {t}'))

            dataManager.stopAnimation()

            resultOja = oja.forward(
                data=Xc[i][0], 
                iter=20000, 
                asyn=False, 
                print=lambda c, t: dataManager.show([dataManager.resize(c)], f'Oja iteration {t}'))

            dataManager.stopAnimation()
        else:
            dataManager.show([dataManager.resize(Xc[i][0])], f'Orginal')
            dataManager.stopAnimation()

            resultOja = oja.forward(
                data=Xc[i][0], 
                iter=20000, 
                asyn=True)
            dataManager.show([dataManager.resize(resultOja)], f'Oja final')
            dataManager.stopAnimation()

            resultHebb = hebb.forward(
                data=Xc[i][0], 
                iter=20000, 
                asyn=True)
            dataManager.show([dataManager.resize(resultOja)], f'Hebb final')
            dataManager.stopAnimation()

        print(f'sample: {i+1}\tHebb: {np.all(resultHebb==X[i])}\tOja: {np.all(resultOja==X[i])}')
        if np.all(resultHebb==X[i]):
            correctHebb += 1
        
        if np.all(resultOja==X[i]):
            correctOja +=1

    print(f'Oja learning rule got {correctOja}/{len(X)} right')
    print(f'Hebb learning rule got {correctHebb}/{len(X)} right')

if __name__ == "__main__":
    # test0()
    testAccuracyOfTrainingMethods()
