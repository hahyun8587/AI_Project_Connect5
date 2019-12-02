import Agent as agent
import Environment as env
import utils
import numpy as np

def step(envir, modelA, modelB): 
    merit = 10
    demerit = -10

    state = envir.state
    policy = modelA.predict(state.reshape(-1, 1))
    #print(modelA.A[-1][-1], modelA.A[-1][-1] - modelA.A[-1][-1].max(axis = 0), modelA.softmax(modelA.A[-1][-1]), modelA.W[-1], sep = "\n") #for debug
    #action = modelA.act(policy) 
    reward = envir.reward(modelA.act(policy), modelA.color)
    
    modelA.saveSample(state, policy, reward)

    print(reward)

    if reward == merit:
        modelB.samples[-1][2] = demerit

        return False
    elif reward == demerit:
        modelB.samples[-1][2] = merit

        return False

    return True    

dnA = "modelA_weights.txt"
dnB = "modelB_weights.txt"
modelA = None
modleB = None
load = True
epoch = 10000
eta = 0.01
gamma = 0.9
seed = 7
inp = 225
hidden = 300
outp = 225
num = 10
size = 15
goal = 5

arch = utils.netArch(inp, hidden, outp, num)
modelA = agent.Agent(arch, eta, gamma, utils.load(dnA, arch) if load else None)
modelB = agent.Agent(arch, eta, gamma, utils.load(dnB, arch) if load else None)
envir = env.Environment(size, goal)

"""
modelA.color = -1
modelB.color = 1

for i in range(epoch):
    print(i)

    while 1:
        print("A")
        if not step(envir, modelA, modelB):
            break
        print("B")
        if not step(envir, modelB, modelA):
            break

    envir.show()    
    modelA.optimizeEp()
    modelB.optimizeEp()
    modelA.clear()
    modelB.clear()
    envir.clear()
"""

for i in range(epoch):
    black = -1
    white = 1

    if np.random.random() < 0.5:
        modelA.color = black
        modelB.color = white
    else:
        modelA.color = white
        modelB.color = black   
    
    while 1: 
        if not step(envir, modelA, modelB):
            break
        
        envir.show()  

        if not step(envir, modelB, modelA):
            break

        envir.show()  
    
    envir.show()
    modelA.optimizeEp()
    modelB.optimizeEp()
    modelA.clear()
    modelB.clear()
    envir.clear()    
        
utils.save(dnA, modelA.W)
utils.save(dnB, modelB.W)

        













