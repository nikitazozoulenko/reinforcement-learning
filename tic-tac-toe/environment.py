import copy
import numpy as np


def step(s, a):
    s_prime = s.copy().take_action(a)
    terminate = s_prime.check_win_position()
    if terminate:
        r = 1
    else:
        r = 0
    return s_prime, r, terminate


class GameBoard:
    def __init__(self, size=5, win_length=3):
        self.size = size
        self.win_length = win_length
        self.board = np.zeros(((size, size)), dtype=np.float32)

                
    def take_action(self, a):
        '''Takes action index number a'''
        self.board[a//self.size, a%self.size] = 1
        return self

    
    def check_if_legal_action(self, a):
        if self.board[a//self.size, a%self.size] == 0:
            return True
        return False

    
    def reverse_player_positions(self):
        self.board *= -1
        return self

    
    def copy(self):
        return copy.deepcopy(self)

    
    def restart(self):
        '''Resets board to all zeros'''
        self.board = np.zeros(((self.size, self.size)))
        return self

    
    def check_win_position(self):
        '''Finds the positions of the win (player_val=1). 
           Returns empty list if no win, else list (y,x) coords of positions.'''
        n_cols = np.zeros((self.size)) # |
        n_rows = np.zeros((self.size)) # --
        n_diag = np.zeros((2*self.size-1)) # \
        n_anti_diag = np.zeros((2*self.size-1)) # /

        #enumerate the amount of Xs
        for y, rows in enumerate(self.board):
            for x, value in enumerate(rows):
                if value == 1:
                    n_cols[x] += 1
                    n_rows[y] += 1
                    n_diag[self.size-1-y+x] += 1
                    n_anti_diag[x+y] += 1
        
        #find coords for lists with enough Xs
        for (v,  kind) in [(n_cols, "cols"), (n_rows, "rows"), (n_diag, "diag"), (n_anti_diag, "anti_diag")]:
            for i, val in enumerate(v):
                if val >= self.win_length:
                    return self._find_coords(kind, i)
        return []

        
    def _find_coords(self, kind, index):
        '''Returns a list of (y, x) coordinates of a vector where there exists a win'''
        coords = []
        count = 0

        if kind == "cols":
            for i, val in enumerate(self.board[:, index]):
                if val == 1:
                    count += 1
                    coords += [(index, i)]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []
        
        if kind == "rows":
            for i, val in enumerate(self.board[index, :]):
                if val == 1:
                    count += 1
                    coords += [(i, index)]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []
        
        if kind == "diag":
            for i in range(self.size-abs(index-self.size+1)):
                if self.board[i, i] == 1:
                    count += 1
                    coords += [(i, i)]

                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []

        if kind == "anti_diag":
            for i in range(self.size-abs(index-self.size+1)):
                if index <= self.size-1:
                    coord = (i, self.size-1-abs(index-self.size+1)-i)
                else: #if index > self.size-1
                    coord = (i+abs(index-self.size+1), self.size-1-i)
                if self.board[coord] == 1:
                    count += 1
                    coords += [coord]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []

        
    def __str__(self):
        string = ""
        for rows in self.board:
            for value in rows:
                if value == 1:
                    string += " X " 
                elif value == -1:
                    string += " O "
                else:
                    string += " . "
            string += "\n"
        return string


if __name__ == "__main__":
    env = GameBoard(size=2, win_length=2)
    print(env)
    win_coords = env.check_win_position()
    print(win_coords)