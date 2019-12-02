import numpy as np

class Agent():
    def __init__(self, arch, eta, gamma, W = None, seed = None, epoch = None):
        self.states = []
        self.actions = []
        self.rewards = []
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
                self.W.append(np.array(2 * np.random.random((self.arch[i], self.arch[i - 1])) - 1, dtype = np.float64))

            print(self.W)    
        else:
            self.W = W
    """
    def norm(self, A):
        A_min = A.min(axis = 0)
        A_max = A.max(axis = 0)

        return A - A_min / (A_max - A_min) 
    """
    """
    def stand(self, R):
        return (R - R.mean(axis = 0)) / R.std(axis = 0) 
    """

    def LeakyRelu(self, Z):
        arr = Z.copy()

        for i in range(len(arr)):
            for j in range(len(arr[0])):
                if arr[i][j] <= 0:
                    arr[i][j] *= 0.01

        return arr                

    def LeakyReluPrime(self, Z):
        return np.where(Z > 0, 1, 0.01)  

    def relu(self, Z):
        return Z * (Z > 0)
         
    def reluPrime(self, Z):
        return np.where(Z > 0, 1, 0)   

    def softmax(self, A):
        #print("A", A, sep = "\n")
        norm = A - A.max(axis = 0)
        exps = np.exp(norm)
        
        return exps / exps.sum(axis = 0)
    
    """
    def softmaxPrime(self, A):
        sm = self.softmax(A)

        return sm * (1 - sm)

    def logSoftmax(self, A):
        norm = A - A.max(axis = 0)
        return norm - np.log(np.exp(norm).sum(axis = 0))   
    """

    def predict(self, X):
        Z = []
        A = [X]

        for i in range(len(self.W)):
            Z.append(np.dot(self.W[i], A[i]))    
            A.append(self.relu(Z[i]))
        
        self.A.append(A)
        self.Z.append(Z)

        #print(self.softmax(A[-1]))

        return self.softmax(A[-1]) 

    def act(self, policy):
        return np.random.choice(self.arch[-1], p = policy.flat) 

    def saveSample(self, state, action, reward):
        self.states.append(state)
        
        oneHot = np.zeros(self.arch[-1])
        oneHot[action] = 1
        
        self.actions.append(oneHot)
        self.rewards.append(reward)

    """
    def prob(self):
        return np.array(self.actions) * self.np.array(self.A).T
    """
    
    def calcG_t(self):
        n = len(self.states)

        dp = np.zeros(n)
        dp[n - 1] = self.rewards[n - 1]

        for i in range(n - 2, -1, -1):
            dp[i] = self.rewards[i] + self.gamma * dp[i + 1]

        self.cost.append(dp[0])
        self.g = dp
    
    def g_t(self, t):
        return self.g[t]

    def optimize(self, t):
        g_t = self.g_t(t)
        #reward = self.g.sum(axis = 0)

        for i in range(len(self.W) - 1):
            #policy = np.dot(self.W[i + 1].T, self.softmax(self.A[t][i + 2]))
            delta = self.eta * (np.dot((self.arch[-1] * self.softmax(self.A[t][i + 1]) - 1) * self.reluPrime(self.Z[t][i]), self.A[t][i].T)) * g_t
            self.W[i] += delta
            #print("delta\n", delta)
        self.W[-1] += self.eta * np.dot((self.arch[-1] * self.softmax(self.A[t][-1]) - 1) * self.reluPrime(self.Z[t][-1]), self.A[t][-2].T) * g_t

    def optimizeEp(self):
        n = len(self.states)

        self.calcG_t()

        for i in range(n):
            self.optimize(i)

        self.clear()    
    
    def clear(self):
        for i in range(len(self.states)):
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.Z.pop(0)
            self.A.pop(0)
