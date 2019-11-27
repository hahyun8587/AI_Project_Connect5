import numpy as np

class Environment():
    def __init__(self, size, goal = 5):
        self.state = np.zeros((size, size))
        self.goal = goal
    
    def halfConn(self, action, num, step_j, step_k, color):
        n = len(self.state)
        j = action // n
        k = action % n
        count = 0
        
        for i in range(num - 1):
            j += step_j
            k += step_k

            if j < 0 or j > n - 1 or k < 0 or k > n - 1:
                break

            if self.state[j][k] == color:
                count += 1
            else:
                break    
        
        return count

    def connect(self, action, num, color):
        n = len(self.state)
        
        for i in range(4):
            step_j = -1
            step_k = -1

            if i == 0:
                step_j = 0
            elif i == 2:
                step_k = 0
            elif i == 3:
                step_k = 1 

            if halfConn(action, num, step_j, step_k, color) + halfConn(action, num, -step_j, -step_k, color) >= num - 1:
                return True       

        return False
       

    def halfThree(self, status, end, blank, action, step_j, step_k, color):
        n = len(self.state)
        num = 5
        j = action // n
        k = action % n
        
        for i in range(num):
            j += step_j
            k += step_k

            if j < 0 or j > n - 1 or k < 0 or k > n - 1:
                end[direct] = -color

                if not self.state[j - step_j][k - step_k]:
                    status[1] -= 1
                    blank[direct] = 1

                break

            if self.state[i][j] == color:
                status[0] += 1

                if status[0] > 2 or status[1] == 2:
                    return False    
            elif self.state[i][j] == -color:
                if self.state[j - step_j][k - step_k]:
                    return False
                else:
                    status[1] -= 1
                    blank[direct] = 1
                    end[direct] = self.state[i][j]

                    break    
            else:
                if not self.state[j - step_j][k - step_k]:
                    status[1] -= 1
                    blank[direct] = 2
                    
                    if j + step_j >= 0 and j + step_j < n and k + step_k >= 0 and k + step_k < n:
                        end[direct] = self.state[j + step_j][k + step_k]
                    else:
                        end[direct] = -color

                    break    

                status[1] += 1

        return True

    def check(self, status, end, blank, color):
        if not status[1]:
            if end[0] == color and end[1] == color:
                return False
            elif end[0] == color and end[1] == -color and blank[1] == 1:
                return False
            elif end[0] == -color and blank[0] == 1 and end[1] == color:
                return False
        
        return True    

    def three(self, action, color, flag):
        status = [0 for i in range(2)]
        end = [0 for i in range(2)]
        blank = [0 for i in range(2)]
        step_j = -1
        step_k = -1

        if flag == 0:
            step_j = 0
        elif flag == 2:
            step_k = 0
        elif flag == 3:
            step_k = 1       

        if not halfThree(status, end, blank, action, step_j, step_k, color):
            return False

        if not halfThree(status, end, blank, action, -step_j, -step_k, color):
            return False

        if check(status, end, blank, color):
            return True
        else:
            return False             


    def threes(self, action, color):
        count = 0

        for i in range(4):
            if three(action, i, color):
                count += 1

            if count == 2:
                return True

        return False    

    def move(self, action, color, step_j, step_k, lim):
        n = len(self.state)
        j = action // n
        k = action % n
        count = 0
        
        while 1:
            if self.state[j + step_j][k + step_k] != color:
                break

            j += step_j
            k += step_k
            count += 1

            if count == lim:
                break

        return n * j + k    

    def checkFour(self, color, point, step_j, step_k, lim):
        n = len(self.state)
        blank = 0
        hit = 0
        miss = 0
        point_j = j = point // n
        point_k = k = point % n

        if self.state[point_j + step_j][point_k + step_k] == color:
            return False
        
        for i in range(lim + 1):
            j -= step_j
            k -= step_k

            if self.state[j][k] == color:
                hit += 1
            elif self.state[j][k] == -color:
                miss += 1    
            else:
                blank += 1

        sf = self.state[point_j + 2 * step_j][point_k + 2 * step_k]
        sb = self.sate[point_j + step_j][point_k + step_k]    
        ef = self.state[j][k]
        eb = self.state[j - step_j][k - step_k]
        
        if hit != lim:        
            return False
        elif miss:
            if sb == 0 and sf != color and ef == -color:
                return True
            else:
                return False
        else:
            if ef == 0:
                if (sf != color and sb == 0) or eb != color:
                    return True
                else:
                    return False    
            else:
                if eb != color:
                    return True
                else:
                    return False    

    def four(self, action, color, flag):
        n = len(self.state)
        lim = 3
        step_j = -1
        step_k = -1
        count = 0

        if flag == 0:
            step_j = 0
        elif flag == 2:
            step_k = 0
        elif flag == 3:
            step_k = 1

        if checkFour(color, move(action, color, step_j, step_k, lim), step_j, step_k, lim):
            count += 1
    
        if checkFour(color, move(action , color, -step_j, -step_k, lim), -step_j, -step_k, lim):
            count += 1

        return count         

    def fours(self, action, color):
        side = 4
        count = 0

        for i in range(side):
            count += four(action, color, i)

        if count >= 2:
            return True
        else:
            return False    















    def win(self, action, color):
        if color == -1:
            if not connect(self.goal + 1, action, color) and connect(self.goal, action, color):
                return True
        else:
            if connect(self.goal, action, color):
                return True
        
        return False

    def lose(self, action, color):
        n = len(self.state)

        if self.state[action // n][action % n]:
            return True      
        elif self.color == -1:
            if connect(self.goal + 1, action):
                return True    
            elif threes(action, color):
                return True

        
        
        else:
            return False








    

