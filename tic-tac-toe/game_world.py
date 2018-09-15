import copy
import numpy as np

class GameWorld:
    def __init__(self, size=3, win_length=2):
        self.size = size
        self.win_length = win_length
        self.board = np.zeros(((size, size)))
        self.board[1,2] = 1
        self.board[2,1] = 1
        #self.board[2,2] = 1

    
    def __str__(self):
        return str(self.board)

                
    def take_action(self, a):
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

        #enumerate the amount of X's
        for y, rows in enumerate(self.board):
            for x, value in enumerate(rows):
                if value == 1:
                    n_cols[x] += 1
                    n_rows[y] += 1
                    n_diag[self.size-1-y+x] += 1
                    n_anti_diag[x+y] += 1

        print(n_diag)
        print(n_anti_diag)
        
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
            for i in range(self.size-abs(index%(self.size-1))):
                if self.board[i, i] == 1:
                    count += 1
                    coords += [(i, i)]

                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []

        if kind == "anti_diag":
            for i in range(self.size):
                if index > self.win_length:
                    coord = (i+abs(index%(self.size-1)), self.size-1-i)
                elif index < self.win_length:
                    coord = (i, self.size-1-i+ abs(index%(self.size-1)))
                else:
                    coord = (i, self.size-1-i)
                if self.board[coord] == 1:
                    count += 1
                    coords += [coord]
                    if count == self.win_length:
                        return coords
                else:
                    count = 0
                    coords = []

    

if __name__ == "__main__":
    env = GameWorld()
    print(env)
    win_coords = env.check_win_position()
    print(win_coords)



class Cube():
    def __init__(self):
        self.cube_array = self.init_cube()

        self.vec_to_color = {(1,0,0,0,0,0): "white",
                             (0,1,0,0,0,0): "yellow",
                             (0,0,1,0,0,0): "blue",
                             (0,0,0,1,0,0): "green",
                             (0,0,0,0,1,0): "red",
                             (0,0,0,0,0,1): "orange"}

        self.perm_lookup = {0:"r", 1:"r_prime", 2:"l", 3:"l_prime",
               4:"u", 5:"u_prime", 6:"d", 7:"d_prime",
               8:"f", 9:"f_prime", 10:"b", 11:"b_prime"}

        self.actions = [self.r, self.r_prime, self.l, self.l_prime,
                        self.u, self.u_prime, self.d, self.d_prime,
                        self.f, self.f_prime, self.b, self.b_prime]


    def take_action(self, a):
        self.actions[a]()
        return self
    

    def check_if_solved(self):
        for clr, side in enumerate(self.cube_array):
            for row in side:
                for color in row:
                    if not color[clr]:
                        return False
        return True


    def init_cube(self):
        size = 3
        #color (last dim) order : white, yellow, blue, green, red, orange
        cube = np.zeros((6, size, size, 6), dtype=np.float32)# C, H, W, 6
        for side in range(6):
            for y in range(size):
                for x in range(size):
                    #color is one-hot 6d-vector
                    cube[side, y, x, side] = 1
        return cube


    def _print_side(self, side):
        for row in side:
            for ele in row:
                print(ele, end="\t")
            print()
        print()


    def print(self):
        arr = []
        for side in self.cube_array:
            colors = []
            for row in side:
                row_colors = []
                for onehot in row:
                    row_colors += [str(self.vec_to_color[tuple(onehot)])]
                colors += [row_colors]
            arr += [colors]
        arr = np.array(arr)

        tabs = np.array([["" for _ in range(3)] for _ in range(3)])
        self._print_side(np.concatenate((tabs, arr[0]), axis=1))
        self._print_side(np.concatenate((arr[4], arr[2], arr[5]), axis=1))
        self._print_side(np.concatenate((tabs, arr[1]), axis=1))
        self._print_side(np.concatenate((tabs, arr[3]), axis=1))
        

    def _rotate_only_face(self, side, clockwise=True):
        corners = [np.copy(side[0, 0, :]), np.copy(side[0, -1, :]), 
                np.copy(side[-1, -1, :]), np.copy(side[-1, 0, :])]
        edges = [np.copy(side[0, 1, :]), np.copy(side[1, -1, :]), 
                np.copy(side[-1, 1, :]), np.copy(side[1, 0, :])]
        if clockwise:
            corners = np.concatenate((corners[3:4], corners[0:3]))
            edges = np.concatenate((edges[3:4], edges[0:3]))
        else: #if not clockwise
            corners = np.concatenate((corners[1:], corners[0:1]))
            edges = np.concatenate((edges[1:], edges[0:1]))

        #insert new rotated pieces
        side[0, 0, :] = corners[0]
        side[0, -1, :] = corners[1]
        side[-1, -1, :] = corners[2]
        side[-1, 0, :] = corners[3]
        side[0, 1, :] = edges[0]
        side[1, -1, :] = edges[1]
        side[-1, 1, :] = edges[2]
        side[1, 0, :] = edges[3]
        return side


    def r(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        white =  np.copy(self.cube_array[0,:, -1: ,:])
        blue =   np.copy(self.cube_array[2,:, -1: ,:])
        yellow = np.copy(self.cube_array[1,:, -1: ,:])
        green =  np.copy(self.cube_array[3,:, -1: ,:])
        
        self.cube_array[0,:, -1: ,:] = blue
        self.cube_array[2,:, -1: ,:] = yellow
        self.cube_array[1,:, -1: ,:] = green
        self.cube_array[3,:, -1: ,:] = white

        self.cube_array[5] = self._rotate_only_face(side=self.cube_array[5], clockwise=True)


    def r_prime(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        white =  np.copy(self.cube_array[0,:, -1: ,:])
        blue =   np.copy(self.cube_array[2,:, -1: ,:])
        yellow = np.copy(self.cube_array[1,:, -1: ,:])
        green =  np.copy(self.cube_array[3,:, -1: ,:])
        
        self.cube_array[0,:, -1: ,:] = green
        self.cube_array[2,:, -1: ,:] = white
        self.cube_array[1,:, -1: ,:] = blue
        self.cube_array[3,:, -1: ,:] = yellow

        self.cube_array[5] = self._rotate_only_face(side=self.cube_array[5], clockwise=False)


    def l(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        white =  np.copy(self.cube_array[0,:, :1 ,:])
        blue =   np.copy(self.cube_array[2,:, :1 ,:])
        yellow = np.copy(self.cube_array[1,:, :1 ,:])
        green =  np.copy(self.cube_array[3,:, :1 ,:])
        
        self.cube_array[0,:, :1 ,:] = green
        self.cube_array[2,:, :1 ,:] = white
        self.cube_array[1,:, :1 ,:] = blue
        self.cube_array[3,:, :1 ,:] = yellow

        self.cube_array[4] = self._rotate_only_face(side=self.cube_array[4], clockwise=True)


    def l_prime(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        white =  np.copy(self.cube_array[0,:, :1 ,:])
        blue =   np.copy(self.cube_array[2,:, :1 ,:])
        yellow = np.copy(self.cube_array[1,:, :1 ,:])
        green =  np.copy(self.cube_array[3,:, :1 ,:])
        
        self.cube_array[0,:, :1 ,:] = blue
        self.cube_array[2,:, :1 ,:] = yellow
        self.cube_array[1,:, :1 ,:] = green
        self.cube_array[3,:, :1 ,:] = white

        self.cube_array[4] = self._rotate_only_face(side=self.cube_array[4], clockwise=False)


    def u(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        red =    np.copy(self.cube_array[4, 0, : ,:])
        blue =   np.copy(self.cube_array[2, 0, : ,:])
        orange = np.copy(self.cube_array[5, 0, : ,:])
        green =  np.copy(self.cube_array[3,-1, : ,:])
        
        self.cube_array[4, 0, : ,:] = blue
        self.cube_array[2, 0, : ,:] = orange
        self.cube_array[5, 0, : ,:] = green[::-1, :]
        self.cube_array[3,-1, : ,:] = red[::-1, :]

        self.cube_array[0] = self._rotate_only_face(side=self.cube_array[0], clockwise=True)


    def u_prime(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        red =    np.copy(self.cube_array[4, 0, : ,:])
        blue =   np.copy(self.cube_array[2, 0, : ,:])
        orange = np.copy(self.cube_array[5, 0, : ,:])
        green =  np.copy(self.cube_array[3,-1, : ,:])
        
        self.cube_array[4, 0, : ,:] = green[::-1, :]
        self.cube_array[2, 0, : ,:] = red
        self.cube_array[5, 0, : ,:] = blue
        self.cube_array[3,-1, : ,:] = orange[::-1, :]

        self.cube_array[0] = self._rotate_only_face(side=self.cube_array[0], clockwise=False)


    def d(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        red =    np.copy(self.cube_array[4,-1, : ,:])
        blue =   np.copy(self.cube_array[2,-1, : ,:])
        orange = np.copy(self.cube_array[5,-1, : ,:])
        green =  np.copy(self.cube_array[3, 0, : ,:])
        
        self.cube_array[4,-1, : ,:] = green[::-1, :]
        self.cube_array[2,-1, : ,:] = red
        self.cube_array[5,-1, : ,:] = blue
        self.cube_array[3, 0, : ,:] = orange[::-1, :]

        self.cube_array[1] = self._rotate_only_face(side=self.cube_array[1], clockwise=True)


    def d_prime(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        red =    np.copy(self.cube_array[4,-1, : ,:])
        blue =   np.copy(self.cube_array[2,-1, : ,:])
        orange = np.copy(self.cube_array[5,-1, : ,:])
        green =  np.copy(self.cube_array[3, 0, : ,:])
        
        self.cube_array[4,-1, : ,:] = blue
        self.cube_array[2,-1, : ,:] = orange
        self.cube_array[5,-1, : ,:] = green[::-1, :]
        self.cube_array[3, 0, : ,:] = red[::-1, :]

        self.cube_array[1] = self._rotate_only_face(side=self.cube_array[1], clockwise=False)


    def f(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        white =  np.copy(self.cube_array[0,-1,  :, :])
        orange = np.copy(self.cube_array[5, :,  0, :])
        red =    np.copy(self.cube_array[4, :, -1, :])
        yellow = np.copy(self.cube_array[1, 0,  :, :])

        self.cube_array[0,-1,  :, :] = red[::-1, :]
        self.cube_array[5, :,  0, :] = white
        self.cube_array[4, :, -1, :] = yellow
        self.cube_array[1, 0,  :, :] = orange[::-1, :]

        self.cube_array[2] = self._rotate_only_face(side=self.cube_array[2], clockwise=True)


    def f_prime(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        white =  np.copy(self.cube_array[0,-1,  :, :])
        orange = np.copy(self.cube_array[5, :,  0, :])
        red =    np.copy(self.cube_array[4, :, -1, :])
        yellow = np.copy(self.cube_array[1, 0,  :, :])

        self.cube_array[0,-1,  :, :] = orange
        self.cube_array[5, :,  0, :] = yellow[::-1, :]
        self.cube_array[4, :, -1, :] = white[::-1, :]
        self.cube_array[1, 0,  :, :] = red

        self.cube_array[2] = self._rotate_only_face(side=self.cube_array[2], clockwise=False)


    def b(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        white =  np.copy(self.cube_array[0, 0, :, :])
        orange = np.copy(self.cube_array[5, :,-1, :])
        red =    np.copy(self.cube_array[4, :, 0, :])
        yellow = np.copy(self.cube_array[1,-1, :, :])

        self.cube_array[0, 0, :, :] = orange
        self.cube_array[5, :,-1, :] = yellow[::-1, :]
        self.cube_array[4, :, 0, :] = white[::-1, :]
        self.cube_array[1,-1, :, :] = red

        self.cube_array[3] = self._rotate_only_face(side=self.cube_array[3], clockwise=True)


    def b_prime(self):
        #cube config: cross with white top with blue below white
        #color (last dim) order : white, yellow, blue, green, red, orange
        #white on top with blue facing camera
        white =  np.copy(self.cube_array[0, 0, :, :])
        orange = np.copy(self.cube_array[5, :,-1, :])
        red =    np.copy(self.cube_array[4, :, 0, :])
        yellow = np.copy(self.cube_array[1,-1, :, :])

        self.cube_array[0, 0, :, :] = red[::-1, :]
        self.cube_array[5, :,-1, :] = white
        self.cube_array[4, :, 0, :] = yellow
        self.cube_array[1,-1, :, :] = orange[::-1, :]

        self.cube_array[3] = self._rotate_only_face(side=self.cube_array[3], clockwise=False)

    
    def copy(self):
        return copy.deepcopy(self)

    def shuffle(self, n_moves = 1):
        perms = np.random.random_integers(0, len(self.actions)-1, size=n_moves)
        for i in range(0, n_moves-1):
            anti = -2*(i % 2) + 1
            while perms[i] == perms[i]+anti:
                perms[i+1] = np.random.randint(0, len(self.actions))
            self.actions[perms[i]]()
        self.actions[perms[-1]]()
        return self

