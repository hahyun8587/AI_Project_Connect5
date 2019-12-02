import Agent as agent
import Environment as env
import utils
import numpy as np

def step(envir, modelA, modelB, flag): 
    merit = 1
    demerit = -1

    state = envir.state
    policy = modelA.predict(state.reshape(-1, 1))
    action = modelA.act(policy) 
    #print(modelA.A[-1][-1], modelA.A[-1][-1] - modelA.A[-1][-1].max(axis = 0), modelA.softmax(modelA.A[-1][-1]), modelA.W[-1], sep = "\n") #for debug
    reward = envir.reward(action, modelA.color)
    
    modelA.saveSample(state, action, reward)

    if reward == merit:
        modelB.rewards[-1] = demerit

        if not flag:
            print("modelA won by five")
        else:
            print("modelB won by five")    


        return False
    elif reward == demerit:
        modelB.rewards[-1] = merit

        return False

    return True    

dnA = "./modelA_weights.txt"
dnB = "./modelB_weights.txt"
modelA = None
modleB = None
load = True
epoch = 10000
eta = 0.01
gamma = 0.9
seed = 7
inp = 225
hidden = 30
outp = 225
num = 5
size = 15
goal = 5

arch = utils.netArch(inp, hidden, outp, num)
modelA = agent.Agent(arch, eta, gamma, utils.load(dnA, arch) if load else None)
modelB = agent.Agent(arch, eta, gamma, utils.load(dnB, arch) if load else None)
envir = env.Environment(size, goal)

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
        if not step(envir, modelA, modelB, 0):
            break
        
        if not step(envir, modelB, modelA, 1):
            break

    envir.show()
    modelA.optimizeEp()
    modelB.optimizeEp()
    modelA.clear()
    modelB.clear()
    envir.clear()    
        
utils.save(dnA, modelA.W)
utils.save(dnB, modelB.W)

        













