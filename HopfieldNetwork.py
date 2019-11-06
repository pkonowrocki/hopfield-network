import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class HopefieldNetwork(object):
    def __init__(self, size):
        self.num_neurons = size
        self.W = np.zeros((self.num_neurons, self.num_neurons), dtype = float)
    
    def resetNetwork(self):
        self.W = np.zeros(self.W.shape, dtype = float)

    def trainHebb(self, data):
        self.initialize_for_showing_weights()

        rho = np.sum([np.sum(t) for t in data]) / (len(data) * self.num_neurons)

        for i, ex in enumerate(data):
            t = ex - rho
            self.W += np.outer(t, t)
            self.show_weights(f'Iteration#{i+1}')
        
        self.W -= np.diag(self.W)
        self.W /= len(data)
        self.show_weights('Final', is_final = True)
    
    def _async(self, x, W):
        idx = np.random.randint(0, self.num_neurons) 
        x[idx] = np.sign(np.matmul(W[idx].T, x))
        return x

    def _sync(self, x, W):
        return np.sign(np.matmul(W, x))

    def forward(self, data, iter = 20, asyn = False):
        s = data
        if asyn:
            f = self._async
        else:
            f = self._sync
        
        e = self.energy(s)

        for i in range(iter):
            s = f(s, self.W)
            eNew = self.energy(s)

            if e == eNew:
                return s
            
            e = eNew
        
        return s

    def energy(self, s):
        return -0.5*np.matmul(np.matmul(s, self.W), s)

    def initialize_for_showing_weights(self):
        plt.ion()
        plt.figure(figsize=(6, 5))
        plt.tight_layout()

    def show_weights(self, iteration_name, is_final = False):
        plt.clf()
        plt.title(f'Network Weights - {iteration_name}')
        colors = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(colors)

        if is_final:
            plt.savefig('weights.png')
        plt.show(block = is_final)
        plt.pause(0.1)