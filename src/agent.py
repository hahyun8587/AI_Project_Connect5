import numpy as np

class Agent():
    def __init__(self, arch, eta, gamma, W = None, seed = None, epoch = None):
        self.samples = []
        self.Z = []
        self.A = []
        self.cost = []
        self.arch = arch
        self.epoch = epoch
        self.eta = eta
        self.gamma = gamma

        if W == None:
            self.seed = seed
            self.W = []

            np.random.seed(self.seed)

            for i in range(1, len(self.arch)):
                self.W.append(2 * np.random.random((self.arch[i], self.arch[i - 1])) - 1)
        else:
            self.W = W

    def LeakyRelu(self, Z):
        arr = Z.copy()

        for i in range(len(arr)):
            for j in range(len(arr[0])):
                if arr[i][j] <= 0:
                    arr[i][j] *= 0.01

        return arr                

    def LeakyReluPrime(self, Z):
        return np.where(Z > 0, 1, 0.01)   

    def norm(self, A):
        A_min = A.min(axis = 0)
        A_max = A.max(axis = 0)

        return A - A_min / (A_max - A_min) 

    def SM(self, A):
        norm = A - A.max(axis = 0)
        exps = np.exp(norm)
        
        return exps / exps.sum(axis = 0)
    
    def SMPrime(self, A):
        sm = self.SM(A)

        return sm * (1 - sm)

    def predict(self, X):
        Z = []
        A = [X]

        for i in range(len(self.W)):
            Z.append(np.dot(self.W[i], A[i]))    
            A.append(self.LeakyRelu(Z[i]))
        
        self.A.append(A)
        self.Z.append(Z)

        return self.SM(A[-1]) 

    def act(self, policy):
        return np.random.choice(self.arch[-1], p = policy.flat) 

    def saveSample(self, state, policy, reward):
        self.samples.append([state, policy, reward])
    
    def calcG_t(self):
        n = len(self.samples)

        dp = np.zeros(n)
        dp[n - 1] = self.samples[n - 1][2]

        for i in range(n - 2, -1, -1):
            dp[i] = self.samples[i][2] + self.gamma * dp[i + 1]

        self.cost.append(dp[0])
        self.g = dp

    def g_t(self, t):
        return self.g[t]

    def optimize(self, t):
        g_t = self.g_t(t)

        self.W[-1] += self.eta * np.dot(self.SMPrime(self.A[t][-1]) * self.LeakyReluPrime(self.Z[t][-1]), self.A[t][-2].T) * g_t

        for i in range(len(self.W) - 2, -1, -1):
            policy = np.dot(self.W[i + 1].T, self.SM(self.A[t][i + 2]))
            self.W[i] += self.eta * np.dot(policy * (1 - policy) * self.LeakyReluPrime(self.Z[t][i]), self.A[t][i].T) * g_t


    def optimizeEp(self):
        n = len(self.samples)

        self.calcG_t()

        for i in range(n):
            self.optimize(n - 1 - i)

    def clear(self):
        for i in range(len(self.samples)):
            self.samples.pop(0)
            self.Z.pop(0)
            self.A.pop(0)
