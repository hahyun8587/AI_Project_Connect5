import numpy as np
import src.Environment as env

def translate(string, size):
    return size * (15 - int(string[1:len(string)])) + ord(string[0]) - ord("A")

"""
size = 15
goal = 5
color = -1
merit = 1
demerit = -10
    
envir = env.Environment(size, goal)

while 1:
    envir.show()
    reward = envir.reward(translate(input(), size), color) 
    print("reward:", reward)

    if reward == merit:
        print("win")
        
        break
    elif reward == demerit:
        print("lose")     

        break
"""    