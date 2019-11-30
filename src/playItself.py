import AI_Project_Connect5.src.Agent as agent
import AI_Project_Connect5.src.Environment as env
import AI_Project_Connect5.src.utils as utils
import numpy as np

def step(envir, modelA, modelB): 
    merit = 10
    demerit = -10

    state = envir.state
    policy = modelA.predict(state)
    reward = envir.reward(modelA.act(policy))
    modelA.saveSample(state, policy, reward)

    if reward == merit:
        modelB.samples[-1][2] = demerit

        return False
    elif reward == demerit:
        modelB.samples[-1][2] = merit

        return False

    return True    

modelA = None
modleB = None
load = True
epoch = 1000
eta = 0.01
gamma = 0.9
inp = 225
hidden = 300
outp = 225
num = 10
size = 15
goal = 5

arch = utils.netArch(inp, hidden, outp, num)

if load:
    dnA = "modelA_weights.txt"
    dnB = "modelB_weights.txt"

    modelA = agent.Agent(arch, eta, gamma, utils.load(dnA, arch))
    modelB = agent.Agent(arch, eta, gamma, utils.load(dnB, arch))
else:
    modelA = agent.Agent(arch, eta, gamma)
    modelB = agent.Agent(arch, eta, gamma)

envir = env.Environment(size, goal)

for i in range(epoch):
    black = -1
    white = 1

    if np.random.random() < 0.5:
        modelA.setColor(black)
        modelB.setColor(white)
    else:
        modelA.setColor(white)
        modelB.setColor(black)    

    while 1: 
        if not step(envir, modelA, modelB):
            break
        
        if not step(envir, modelB, modelA):
            break

    modelA.optimizeEp()
    modelB.optimizeEp()    
        
utils.save(dnA, modelA.W)
utils.save(dnB, modelB.W)


        













