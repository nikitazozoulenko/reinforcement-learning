import copy
import numpy as np
import torch


device = torch.device("cuda")

def step(s, a):
    s_prime = s.copy().take_action(a)
    terminate, coords = s_prime.check_win_position()
    if terminate:
        if coords:
            r = float(s_prime.board[coords[0]]) # 1 cross, -1 circle
        else:
            r = 0
    else:
        r = 0
    return s_prime, r, terminate


class GameBoard:
    def __init__(self, size=5, win_length=3):
        self.size = size
        self.win_length = win_length
        self.board = np.zeros(((size, size)), dtype=np.float32)
        self.turn_value_dict = {"X":1, "O":-1}
        self.turn = "X"


    def to_tensor(self):
        '''Returns tensor representation of game board state'''
        tensor = torch.from_numpy(self.board)
        return tensor.to(device).view(1, -1)
    

    def step(self, a):
        s_prime, r, terminate = step(self, a)
        self.take_action(a)
        return r, terminate


    def take_action(self, a):
        '''Takes action index number a'''
        self.board[a//self.size, a%self.size] = self.turn_value_dict[self.turn]
        if self.turn == "X":
            self.turn = "O"
        else:
            self.turn = "X"
        return self

    
    def check_if_legal_action(self, a):
        if self.board[a//self.size, a%self.size] == 0:
            return True
        return False

    
    def get_allowed_actions(self):
        '''Returns np.array of size (-1). True == allowed'''
        return (self.board.reshape(-1) == 0).astype(np.uint8)

    
    def reverse_player_positions(self):
        self.board *= -1
        if self.turn == "X":
            self.turn = "O"
        else:
            self.turn = "X"
        return self

    
    def copy(self):
        return copy.deepcopy(self)

    
    def reset(self):
        '''Resets board to all zeros'''
        self.board = np.zeros(((self.size, self.size)))
        self.turn = "X"
        return self


    def check_win_position(self):
        '''Checks if any player has won and returns list [True/False, coords]'''
        terminate, coords = self._check_win_for_white(self.board)
        if terminate:
            return terminate, coords
        terminate, coords = self._check_win_for_white(-1 * self.board)
        return terminate, coords


    def _check_win_for_white(self, board):
        '''Finds the positions of the win for "X" (player_val=1). 
           Returns tuple (terminate, coord_list), empty list if no win, else list (y,x) coords of positions.'''
        n_cols = np.zeros((self.size)) # |
        n_rows = np.zeros((self.size)) # --
        n_diag = np.zeros((2*self.size-1)) # \
        n_anti_diag = np.zeros((2*self.size-1)) # /
        non_zeros = 0

        #enumerate the amount of Xs
        for y, rows in enumerate(board):
            for x, value in enumerate(rows):
                if value != 0:
                    non_zeros += 1
                if value == 1:
                    n_cols[x] += 1
                    n_rows[y] += 1
                    n_diag[self.size-1-y+x] += 1
                    n_anti_diag[x+y] += 1
        
        #find coords for lists with enough Xs
        for (v,  kind) in [(n_cols, "cols"), (n_rows, "rows"), (n_diag, "diag"), (n_anti_diag, "anti_diag")]:
            for i, val in enumerate(v):
                if val >= self.win_length:
                    coords = self._find_coords(kind, i, board)
                    if coords:
                        return True, coords

        #look for draw
        if non_zeros == self.size**2:
            return True, []

        return False, []

        
    def _find_coords(self, kind, index, board):
        '''Returns a list of (y, x) coordinates of a vector where there exists a win'''
        coords = []
        count = 0

        if kind == "cols":
            for i, val in enumerate(board[:, index]):
                if val == 1:
                    count += 1
                    coords += [(i, index)]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []
        
        elif kind == "rows":
            for i, val in enumerate(board[index, :]):
                if val == 1:
                    count += 1
                    coords += [(index, i)]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []
        
        elif kind == "diag":
            for i in range(self.size-abs(index-self.size+1)):
                if index <= self.size-1:
                    coord = (self.size-1-i, self.size-1-abs(index-self.size+1)-i)
                else: #if index > self.size-1
                    coord = (self.size-1-(i+abs(index-self.size+1)), self.size-1-i)
                if board[coord] == 1:
                    count += 1
                    coords += [coord]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []

        elif kind == "anti_diag":
            for i in range(self.size-abs(index-self.size+1)):
                if index <= self.size-1:
                    coord = (i, self.size-1-abs(index-self.size+1)-i)
                else: #if index > self.size-1
                    coord = (i+abs(index-self.size+1), self.size-1-i)
                if board[coord] == 1:
                    count += 1
                    coords += [coord]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []
        return []

        
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