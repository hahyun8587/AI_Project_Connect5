import numpy as np

class Agent():
    def __init__(self, arch, epoch, eta, seed, gamma, color, W = None):
        self.samples = []
        self.W = []
        self.Z = []
        self.A = []
        self.cost = []
        self.arch = arch
        self.epoch = epoch
        self.eta = eta
        self.seed = seed 
        self.gamma = gamma
        self.color = color

        if W == None:
            np.random.seed(self.seed)

            for i in range(1, len(self.arch)):
                self.W.append(np.random.random((self.arch[i], self.arch[i - 1])))
        else:
            self.W = W

    def relu(self, Z):
        return max(0, Z)

    def reluPrime(self, Z):
        return np.where(Z <= 0, 0, 1)   

    def SM(self, A):
        norm = A - A.max(axis = 0)
        exps = np.exp(norm)
        
        return exps / exps.sum(axis = 0)
    
    def SMPrime(self, A):
        sm = self.SM(A)

        return sm * (1 - sm)

    def predict(self, X):
        A = [X]
        Z = []

        for i in range(len(self.W)):
            Z.append(np.dot(self.W[i], A[i]))    
            A.append(self.relu(Z[i]))
        
        return Z, A, self.SM(A[-1]) 

    def act(self, policy):
       return np.random.choice(self.arch[-1], p = policy.flat) 

    def saveSample(self, state, Z, A, policy, reward):
        self.samples.append((state, policy, reward))
        self.A.append(A)
        self.Z.append(Z)

    def calcG_t(self):
        n = len(self.samples)

        dp = np.array(n)
        dp[n - 1] = self.samples[n - 1][2]

        for i in range(n - 2, -1, -1):
            dp[i] = self.samples[i][2] + self.gamma * dp[i + 1]

        self.cost.append(dp[0])
        self.g = dp

    def g_t(self, t):
        return self.g[t]

    def optimize(self, t):
        g_t = self.g_t(t)

        self.W[-1] += self.eta * np.dot(self.SMPrime(self.A[t][-1]) * self.reluPrime(self.Z[t][-1]), self.A[t][-2].T) * g_t

        for i in range(len(self.W) - 2, -1, -1):
            policy = np.dot(self.W[i + 1].T, self.SM(self.A[t][i + 2]))
            self.W[i] += self.eta * np.dot(policy * (1 - policy) * self.reluPrime(self.Z[t][i]), self.A[t][i].T) * g_t

    def clear(self):
        for i in range(len(self.samples)):
            self.samples.pop(0)
            self.Z.pop(0)
            self.A.pop(0)    

