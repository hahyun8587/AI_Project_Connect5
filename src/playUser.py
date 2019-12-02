import Agent as agent
import Environment as env
import utils
import numpy as np

def translate(string, size):
    return size * (int(string[0:len(string) - 1]) - 1) + ord(string[len(string) - 1]) - ord("A")

def step(envir, model, flag):
    merit = 1
    demerit = -1

    if not flag:
        reward = envir.reward(translate(input("your turn: "), envir.size), -model.color)
        envir.show()

        if reward == merit:
            print("you win")
            
            return False
        elif reward == demerit:
            print("you lose")

            return False

        reward = envir.reward(model.act(model.predict(envir.state.reshape(-1, 1))), model.color)
        envir.show()

        if reward == merit:
            print("you lose")

            return False
        elif reward == demerit:
            print("you win")

            return False 
    else:
        reward = envir.reward(model.act(model.predict(envir.state.reshape(-1, 1))), model.color)
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
        
dn = "modelA_weights.txt"
eta = 0.01
gamma = 0.9
inp = 225
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

envir.show()    
    
while 1:
    if color == black:
        if not step(envir, model, 0):
            break
    else:
        if not step(envir, model, 1):
            break    
        







