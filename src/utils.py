import numpy as np

def check(lines, row, col):
    if len(lines) != row:
        return False

    for i in range(len(lines)):
        count = 0

        for j in range(len(lines[i])):
            if lines[i][j] == ' ':
                count += 1

        if count + 1 != col:
            return False

    return True        

#load weights
def load(dn, arch):
    try:
        fp = open(dn, "r")
        lines = fp.readlines()

        part = []
        j = 0

        for i in range(len(lines)):
            if lines[i][0] == '\n':
                if not check(part, arch[j + 1], arch[j]):
                    print("wrong net arch")
                    
                    return None

                part = []
                j += 1  
            else:
                part.append(lines[i])   

        W = []
        wStr = []
        nStr = []

        for i in range(len(lines)):
            if lines[i][0] == '\n':
                wStr.append(nStr)

                nStr = []
            else:
                nStr.append(lines[i][:-1].split(" "))
        
        fp.close()        
    except IOError:
        print("no file found in the directory")            

    for i in range(1, len(arch)):
        W.append(np.zeros((arch[i], arch[i - 1]), dtype = np.float64))

    for i in range(len(W)):
        for j in range(len(W[i])):
            for k in range(len(W[i][0])):
                W[i][j][k] = int(wStr[i][j][k]) if wStr[i][j][k].isdigit() else float(wStr[i][j][k])

    return W            
'''
@param 
dn: directory name
arch: net arch

@return 
weights of the nerual network
returns None if the net arch is not matched with weights
'''

#saves weights to the file
def save(dn, W):
    fp = open(dn, "w")

    for i in range(len(W)):
        for j in range(len(W[i])):
            string = ""

            for k in range(len(W[i][0])):
                string += str(W[i][j][k])
                string += " " if k < len(W[i][0]) - 1 else "\n"

            fp.write(string)

        fp.write("\n")    

    fp.close()
'''
@param
dn: directory name
W: list of weights
'''

def netArch(inp, hidden, outp, num):
    arch = []

    arch.append(inp)

    for i in range(num):
        arch.append(hidden)

    arch.append(outp)

    return arch    


        
