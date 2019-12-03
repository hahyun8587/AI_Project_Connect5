import numpy as np

class Environment():
    def __init__(self, size, goal = 5):
        self.state = np.zeros((size, size), dtype = int)
        self.size = size
        self.goal = goal

    def place(self, action, color):
        n = len(self.state)
        self.state[action // n][action % n] = color

    def outOfRange(self, idx_i, idx_j):
        n = len(self.state)

        if idx_i < 0 or idx_i > n - 1 or idx_j < 0 or idx_j > n - 1:
            return True

    def halfConn(self, action, num, step_j, step_k, color):
        n = len(self.state)
        j = action // n
        k = action % n
        count = 0
                
        for i in range(num - 1):
            j += step_j
            k += step_k

            if self.outOfRange(j, k):
                break

            if self.state[j][k] == color:
                count += 1
            else:
                break    
    
        return count

    def connect(self, action, num, color):
        n = len(self.state)
        side = 4

        for i in range(side):
            step_j = -1
            step_k = -1

            if i == 0:
                step_j = 0
            elif i == 2:
                step_k = 0
            elif i == 3:
                step_k = 1 

            if self.halfConn(action, num, step_j, step_k, color) + self.halfConn(action, num, -step_j, -step_k, color) >= num - 1:
                return True       

        return False
   
    def move(self, action, step_j, step_k, lim, color):
        n = len(self.state)
        j = action // n
        k = action % n
        count = 0
        
        while 1:
            if self.outOfRange(j + step_j, k + step_k):
                break
            
            if self.state[j + step_j][k + step_k] != color:
                break

            j += step_j
            k += step_k
            count += 1

            if count == lim:
                break

        return n * j + k    

    def addList(self, lst, num, point, step_j, step_k, color):
        n = len(self.state)
        j = point // n
        k = point % n

        for i in range(num):
            if self.outOfRange(j, k):
               lst.append(-color)
            else:    
                lst.append(self.state[j][k])

            j += step_j
            k += step_k    

    def checkThree(self, point, step_j, step_k, lim, color):
        #print("\n[searching] step j:", step_j, "step_k:", step_k)
        n = len(self.state)
        hit = 0
        miss = 0
        blank = 0
        point_j = j = point // n
        point_k = k = point % n
        s = []
        e = []
        num = 3

        if not self.outOfRange(point_j - step_j, point_k - step_k):            
            if self.state[point_j - step_j][point_k - step_k]:
                return False
        else:
            return False        
        
        for i in range(lim):
            j += step_j
            k += step_k

            if self.outOfRange(j, k):
                return False

            if self.state[j][k] == color:
                hit += 1
            elif self.state[j][k] == -color:
                miss += 1
            else:
                blank += 1        

        self.addList(s, num - 1, n * (point_j - num * step_j) + point_k - num * step_k, step_j, step_k, color)
        self.addList(e, num + 1, n * j + k, step_j, step_k, color)

        #print(hit, miss, blank, lim)
        #print(s, e, sep = '\n')

        if hit != lim - 1:      # 2 hits
            #print("not forming three(no 3 hits)")
            return False
        elif miss:          # no miss
            #print("not forming three(has miss)")
            return False
        else:
            if e[0] == 0:   # type1(all connected)
                if e[1] == color:   # 1 blank 1 hit
                    #print("not forming three(type1: 1b 1h)")
                    return False
                elif (e[1] == 0 and e[2] == color) or e[1] == -color:   # 2 blanks 1 hit or 1 blank 1 miss
                    #print("forming three(type1: 2b 1h or 1 1b 1m)")
                    if s[0] != color and s[1] == 0:
                        return True
                    else:
                        return False
                elif e[1] == 0 and e[2] != color:   # 2 blanks 1 miss or 3 blanks
                    #print("not forming three(type1: 2b 1m or 3b)")
                    if s[1] == color:
                        return False
                    else:
                        return True             
            else:           # type2(1 blank inside)       
                if e[1] != 0:   # no blank
                    #print("not forming three(type2: 0b)") 
                    return False
                else:
                    if e[2] == color:   # 1 blank 1 hit 
                        #print("not forming three(type2: 1b 1h)") 
                        return False
                    elif e[2] == 0 or e[2] == -color:   # 2 blanks or 1 blank 1 miss 
                        #print("not forming three(type2: 2b or 1b 1m)") 
                        if s[1] == color:
                            return False     
                        else:
                            return True        

    def three(self, action, color, flag):
        lim = 2
        step_j = -1
        step_k = -1

        if flag == 0:
            step_j = 0
        elif flag == 2:
            step_k = 0
        elif flag == 3:
            step_k = 1

        #print("moved point:", self.move(action, step_j, step_k, lim, color))
        
        if self.checkThree(self.move(action, step_j, step_k, lim, color), -step_j, -step_k, lim + 1, color):
            return True
        elif self.checkThree(self.move(action, -step_j, -step_k, lim, color), step_j, step_k, lim + 1, color):
            return True
        else:
            return False        
        
    def threes(self, action, color):
        side = 4
        count = 0

        for i in range(side):
            if self.three(action, color, i):
                count += 1
        #print("threes:", count)
        if count >= 2:
            return True
        else:
            return False            


    def checkFour(self, point, step_j, step_k, lim, color, found):
        n = len(self.state)
        hit = 0
        miss = 0
        blank = 0
        point_j = j = point // n
        point_k = k = point % n
        s = []
        e = []
        num = 2

        if not self.outOfRange(point_j - step_j, point_k - step_k):
            if self.state[point_j - step_j][point_k - step_k] == color:
                return False
        
        for i in range(lim):
            j += step_j
            k += step_k

            if self.outOfRange(j, k):
                return False

            if self.state[j][k] == color:
                hit += 1
            elif self.state[j][k] == -color:
                miss += 1    
            else:
                blank += 1

        self.addList(s, num, n * (point_j - num * step_j) + point_k - num * step_k, step_j, step_k, color)
        self.addList(e, num, n * j + k, step_j, step_k, color)
       
        if hit != lim - 1:    
            #print("not forming four(no 3 hits)")    
            return False
        elif miss:
            if found:
                return False

            if s[1] == 0 and s[0] != color and e[0] == -color:
                #print("forming four(type1: 1m)")
                return True
            else:
                return False
        else:
            if e[0] == 0:
                if found:
                    return False

                if (s[0] != color and s[1] == 0) or e[1] != color:
                    #print("forming four(type1: 1b no hit)")
                    return True
                else:
                    return False    
            else:
                if e[1] != color:
                    #print("forming four(type2: no hit)")
                    return True
                else:
                    return False    

    def four(self, action, color, flag):
        n = len(self.state)
        lim = 3
        step_j = -1
        step_k = -1
        count = 0
        found = False

        if flag == 0:
            step_j = 0
        elif flag == 2:
            step_k = 0
        elif flag == 3:
            step_k = 1

        if self.checkFour(self.move(action, step_j, step_k, lim, color), -step_j, -step_k, lim + 1, color, found):
            count += 1
            found = True
    
        if self.checkFour(self.move(action, -step_j, -step_k, lim, color), step_j, step_k, lim + 1, color, found):
            count += 1

        return count         

    def fours(self, action, color):
        side = 4
        count = 0

        for i in range(side):
            count += self.four(action, color, i)
        #print("fours:", count)
        if count >= 2:
            return True
        else:
            return False    

    def win(self, action, color):
        if color == -1:
            if not self.connect(action, self.goal + 1, color) and self.connect(action, self.goal, color):
                return True
        else:
            if self.connect(action, self.goal, color):
                return True
        
        return False

    def lose(self, action, color):
        n = len(self.state)

        if self.state[action // n][action % n]:
            return True      
        
        self.place(action, color)

        if color == -1:
            if self.connect(action, self.goal + 1, color):
                return True    
            elif self.threes(action, color):
                return True
            elif self.fours(action, color):
                return True    
        
        return False

    def reward(self, action, color):
        merit = 1
        demerit = -10
        neut = 0

        if self.lose(action, color):
            return demerit
        elif self.win(action, color):
            return merit
        else:
            return neut             

    def clear(self):
        n = len(self.state)

        for i in range(n):
            for j in range(n):
                self.state[i][j] = 0

    def show(self):
        n = len(self.state)

        for i in range(n):
            print(-i + self.size, end = " " if -i + self.size >= 10 else "  ")

            for j in range(n):
                print(self.state[i][j], end = "  " if self.state[i][j] >= 0 else " ")

            print()     

        print("   ", end = "")

        for i in range(self.size):
            print(chr(i + 65), end = "  ")

        print("\n")    








    

