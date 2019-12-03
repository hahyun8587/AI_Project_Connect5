import Agent as agent
import Environment as env
import utils
import numpy as np

def translate(string, size):
    return size * (15 - int(string[1:len(string)])) + ord(string[0]) - ord("A")

def step(envir, model, flag):
    merit = 1
    demerit = -10

    if not flag:
        reward = envir.reward(translate(input("your turn: "), envir.size), -model.color)
        envir.show()

        if reward == merit:
            print("you win")
            
            return False
        elif reward == demerit:
            print("you lose")

            return False

        print(model.predict(np.vstack(([1], envir.state.reshape(-1, 1)))))
        action = model.act(model.predict(np.vstack(([1], envir.state.reshape(-1, 1)))))
        reward = envir.reward(action, model.color)
        envir.show()

        if reward == merit:
            print("you lose")

            return False
        elif reward == demerit:
            print("you win")

            return False 
    else:
        print(model.predict(np.vstack(([1], envir.state.reshape(-1, 1)))))
        action = model.act(model.predict(np.vstack(([1], envir.state.reshape(-1, 1)))))
        reward = envir.reward(action, model.color)
        envir.show()

        if reward == merit:
            print("you lose")

            return False
        elif reward == demerit:
            print("you win")

            return False        
                       
        reward = envir.reward(translate(input("your turn: "), envir.size), -model.color)
        envir.show()

        if reward == merit:
            print("you win")
            
            return False
        elif reward == demerit:
            print("you lose")

            return False

    return True
        
dn = "./modelA_weights.txt"
eta = 0.01
gamma = 0.9
inp = 226
hidden = 30
outp = 225
num = 5
size = 15
goal = 5

arch = utils.netArch(inp, hidden, outp, num)
model = agent.Agent(arch, eta, gamma, utils.load(dn, arch))
envir = env.Environment(size, goal)

black = -1
white = 1
color = 0

ch = input("first or second(f/s): ")

if ch == "f":
    color = black
    model.color = 1
else:
    color = white
    model.color = -1 

print()
envir.show()    
    
while 1:
    if color == black:
        if not step(envir, model, 0):
            break
    else:
        if not step(envir, model, 1):
            break    
        







