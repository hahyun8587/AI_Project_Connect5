import numpy as np
import src.Environment as env

size = 15
goal = 5
color = -1
merit = 10
demerit = -10
    
envir = env.Environment(size, goal)

while 1:
    envir.show()
    reward = envir.reward(int(input()), color) 
    print("reward:", reward)

    if reward == merit:
        print("win")
        
        break
    elif reward == demerit:
        print("lose")     

        break
        