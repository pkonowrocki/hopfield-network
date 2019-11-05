import numpy as np

class HopefieldNetwork(object):
    def __init__(self, size):
        self.num_neurons = size
        self.W = np.zeros((self.num_neurons, self.num_neurons), dtype = float)
    
    def resetNetwork(self):
        self.W = np.zeros(self.W.shape, dtype = float)

    def trainHebb(self, data):
        rho = np.sum([np.sum(t) for t in data]) / (len(data) * self.num_neurons)
        for ex in data:
            t = ex - rho
            self.W += np.outer(t, t)
        
        self.W -= np.diag(self.W)
        self.W /= len(data)
    
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