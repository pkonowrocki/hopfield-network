import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class HopefieldNetwork(object):
    def __init__(self, size):
        self.num_neurons = size
        self.W = np.ones((self.num_neurons, self.num_neurons), dtype = float)
    
    def resetNetwork(self):
        self.W = np.ones((self.num_neurons, self.num_neurons), dtype = float)
    
    def trainHebb(self, data, showWeights = False):
        if(showWeights):
            self.initialize_for_showing_weights()

        rho = np.sum([np.sum(t) for t in data]) / (len(data) * self.num_neurons)

        for i, ex in enumerate(data):
            t = ex - rho
            self.W += np.outer(t, t)

            if(showWeights):
                self.show_weights(f'Iteration#{i+1}')
        
        self.W -= np.diag(np.diag(self.W))
        self.W = self.W/len(data)

        if(showWeights):
                self.show_weights(f'Final', is_final=True)

    def trainOja(self, data, u = 0.0001, iter = 1000, showWeights = False):
        self.W = np.random.normal(scale=0.25, size=self.W.shape)
        self.W = (self.W + self.W.T)/2

        if(showWeights):
            self.initialize_for_showing_weights()
            self.show_weights(f'Iteration#{0}')

        for i in range(iter):
            Wprev = self.W.copy()
            for x in data:
                Ys = np.dot(x, self.W)
                t1 = np.outer(Ys,x)
                t2 = np.square(Ys)*self.W
                self.W += u*(t1-t2).reshape(self.W.shape)

            if(showWeights):
                self.show_weights(f'Iteration#{i+1}')
                print(i+1, "\t", np.linalg.norm(Wprev - self.W))

            if np.linalg.norm(Wprev - self.W) < 1e-10:
                break 
        
        self.W -= np.diag(np.diag(self.W))
        if(showWeights):
            self.show_weights(f'Final')

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
        plt.figure(1)
        plt.clf()
        plt.title(f'Network Weights - {iteration_name}')
        colors = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(colors)

        if is_final:
            plt.savefig('weights.png')
        plt.show(block = is_final)
        plt.pause(0.1)