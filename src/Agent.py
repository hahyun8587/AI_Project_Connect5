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
        else:
            self.W = W
    """
    def norm(self, A):
        A_min = A.min(axis = 0)
        A_max = A.max(axis = 0)

        return A - A_min / (A_max - A_min) 
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
    """

    def relu(self, Z):
        return Z * (Z > 0)
         
    def reluPrime(self, Z):
        return np.where(Z > 0, 1, 0)   

    def softmax(self, A):
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

    def objectiveFunc(self, Y, prob):    
        return (np.log((Y * prob).sum(axis = 0)) * self.g).sum(axis = 0)

    def predict(self, X):
        Z = []
        A = [X]

        for i in range(len(self.W)):
            Z.append(np.dot(self.W[i], A[i]))  
            A.append(self.relu(Z[i]))
        
        self.A.append(A)
        self.Z.append(Z)

        return self.softmax(A[-1]) 

    def act(self, policy):
        return np.random.choice(self.arch[-1], p = policy.flat) 

    def saveSample(self, state, action, reward):
        self.states.append(state)
        
        oneHot = np.zeros(self.arch[-1])
        oneHot[action] = 1
        
        self.actions.append(oneHot)
        self.rewards.append(reward)

    def calcG_t(self):
        n = len(self.states)

        dp = np.zeros(n)
        dp[n - 1] = self.rewards[n - 1]

        for i in range(n - 2, -1, -1):
            dp[i] = self.rewards[i] + self.gamma * dp[i + 1]

        #self.g = self.stand(dp)
        self.g = dp
    
    def g_t(self, t):
        return self.g[t]   

    def optimize(self):
        Z = [np.hstack([self.Z[j][i] for j in range(len(self.Z))]) for i in range(len(self.Z[0]))]
        A = [np.hstack([self.A[j][i] for j in range(len(self.A))]) for i in range(len(self.A[0]))]
        Y = np.array(self.actions).T

        self.calcG_t()
        self.cost.append(self.objectiveFunc(Y, self.softmax(A[-1])))
        print("cost: ", self.cost) 
        print("discounted reward:", self.g)
        print("log prob:", np.log((Y * self.softmax(A[-1])).sum(axis = 0)))   
        #print("\n", Z, A, sep = "\n")

        E = Y - self.softmax(A[-1])
        
        self.W[-1] += np.dot(self.g * E * self.reluPrime(Z[-1]), A[-2].T)

        for i in range(len(self.W) - 2, -1, -1):
            E = np.dot(self.W[i + 1].T, E)
            delta = np.dot(self.g * E * self.reluPrime(Z[i]), A[i].T)
            self.W[i] += delta
            #print("delta\n", delta)

        self.clear()    

    """
    def optimizeEp(self):
        n = len(self.states)

        self.calcG_t()
        self.stand(self.g)

        for i in range(n):
            self.optimize(i)

        self.clear()    
    """    
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.Z = []
        self.A = []
    