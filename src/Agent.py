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

        self.B = []

        for i in range(1, len(self.arch)):
            self.B.append(np.zeros((self.arch[i], 1)))

        if W == None:
            self.seed = seed
            self.W = []

            np.random.seed(self.seed)

            for i in range(1, len(self.arch)):
                #self.W.append(np.random.normal(scale = np.sqrt(2 / (self.arch[i - 1] + self.arch[i])), size = (self.arch[i], self.arch[i - 1])))   
                self.W.append(2 * np.random.random((self.arch[i], self.arch[i - 1])) - 1) 
        else:
            self.W = W
        
        #print(self.W)
    """
    def norm(self, A):
        A_min = A.min(axis = 0)
        A_max = A.max(axis = 0)

        return A - A_min / (A_max - A_min) if A_max - A_min else A
    """

    def stand(self, R):
        return (R - R.mean(axis = 0)) / R.std(axis = 0) if R.std(axis = 0) else R
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

    def objectiveFunc(self, Y, probs):    
        return -(np.log((Y * probs).sum(axis = 0)) * self.g).sum(axis = 0)

    def predict(self, X):
        Z = []
        A = [X]

        for i in range(len(self.W) - 1):
            Z.append(np.dot(self.W[i], A[i])) #+ self.B[i])
            #print(Z[i])  
            A.append(self.relu(Z[i]))    
        
        Z.append(np.dot(self.W[-1], A[-1])) #+ self.B[-1])

        self.Z.append(Z)
        self.A.append(A)
        
        return self.softmax(Z[-1]) 

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
    
    def optimize(self):
        n = len(self.W)
        Z = [np.hstack([self.Z[j][i] for j in range(len(self.Z))]) for i in range(len(self.Z[0]))]
        A = [np.hstack([self.A[j][i] for j in range(len(self.A))]) for i in range(len(self.A[0]))]
        Y = np.array(self.actions).T
        dW = []

        #print(Z)
        #print(A)

        self.calcG_t()
        self.cost = []
        self.cost.append(self.objectiveFunc(Y, self.softmax(Z[-1])))
        print("cost:", self.cost) 
        #print("discounted reward:", self.g)
        #print("log prob:", np.log((Y * self.softmax(A[-1])).sum(axis = 0)))   
        #print("prob:", self.softmax(A[-1]))
        #print("\n", Z, sep = "\n")

        #print("A:", A[-1])
        print("policy:", self.softmax(np.array(Z[-1])))
        #print("Y:", Y)

        dZ = self.g * (self.softmax(Z[-1]) - Y)

        dW.append(np.dot(dZ, A[-1].T))
        
        #self.B[-1] -= self.eta * dZ.sum(axis = 1, keepdims = True)

        for i in range(n - 2, -1, -1):
            dZ = np.dot(self.W[i + 1].T, dZ) * self.reluPrime(Z[i]) 
            
            dW.append(np.dot(dZ, A[i].T))
            
            #self.B[i] -= self.eta * dZ.sum(axis = 1, keepdims = True)

        for i in range(n - 1, -1, -1):
            self.W[i] -= self.eta * dW[-i + n - 1]        

        self.clear()    
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.Z = []
        self.A = []
    