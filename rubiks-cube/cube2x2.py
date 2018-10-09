import copy
import numpy as np
import torch


device = torch.device("cuda")

def cube_to_tensor(cube):
    tensor = torch.from_numpy(cube.get_cube_symmetries())
    return tensor.to(device).view(24, -1)


def step(s, a):
    s_prime = s.copy().take_action(a)
    terminate = s_prime.check_if_solved()
    if terminate:
        r = 22
    else:
        r = -1
    return s_prime, r, terminate


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
        for side in self.cube_array:
            clr = np.nonzero(side[0,0])
            for row in side:
                for color in row:
                    if np.nonzero(color) != clr:
                        return False
        return True


    def init_cube(self):
        size = 2
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

        tabs = np.array([["" for _ in range(2)] for _ in range(2)])
        self._print_side(np.concatenate((tabs, arr[0]), axis=1))
        self._print_side(np.concatenate((arr[4], arr[2], arr[5]), axis=1))
        self._print_side(np.concatenate((tabs, arr[1]), axis=1))
        self._print_side(np.concatenate((tabs, arr[3]), axis=1))
        

    def _rotate_only_face(self, side, clockwise=True):
        corners = [np.copy(side[0, 0, :]), np.copy(side[0, -1, :]), 
                np.copy(side[-1, -1, :]), np.copy(side[-1, 0, :])]
        if clockwise:
            corners = np.concatenate((corners[3:4], corners[0:3]))
        else: #if not clockwise
            corners = np.concatenate((corners[1:], corners[0:1]))

        #insert new rotated pieces
        side[0, 0, :] = corners[0]
        side[0, -1, :] = corners[1]
        side[-1, -1, :] = corners[2]
        side[-1, 0, :] = corners[3]
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
        self.random_starting_pos()
        perms = np.random.randint(0, len(self.actions), size=n_moves)
        for i in range(0, n_moves-1):
            anti = -2*(perms[i] % 2) + 1
            while perms[i+1] == perms[i]+anti:
                perms[i+1] = np.random.randint(0, len(self.actions))
            self.actions[perms[i]]()
        if n_moves != 0:
            self.actions[perms[-1]]()
        return self


    def random_starting_pos(self):
        b = []
        g = [self.l, self.l, self.r_prime, self.r_prime]
        w = [self.r_prime, self.l]
        y = [self.r, self.l_prime]
        r = [self.u_prime, self.d]
        o = [self.u, self.d_prime]
        facing_dir = [w, y, b, g, r, o]

        for move in facing_dir[np.random.randint(0,6)]:
            move()
        for _ in range(np.random.randint(0,4)):
            self.f()
            self.b_prime()


    def get_cube_symmetries(self):
        cube = self.copy()

        f0 = [cube.r_prime, cube.l]
        f1 = [cube.r_prime, cube.l]
        f2 = [cube.r_prime, cube.l]
        f3 = [cube.r_prime, cube.l]
        f4 = [cube.u, cube.d_prime]
        f5 = [cube.u, cube.u, cube.d_prime, cube.d_prime]
        rotate_face_perms = [cube.f, cube.b_prime]

        list_arrays = []
        for perms in [f0, f1, f2, f3, f4, f5]:
            for perm in perms:
                perm()
            for _ in range(4):
                for perm in rotate_face_perms:
                    perm()
                list_arrays += [np.copy(cube.cube_array)]
        return np.array(list_arrays)
            


if __name__ == "__main__":
    cube = Cube()
    cube.shuffle(n_moves=0)
    solved = cube.check_if_solved()

    list_arrays = cube.get_cube_symmetries()
    print(list_arrays.shape)

    # for i, arr in enumerate(list_arrays):
    #     if arr.all() == list_arrays[0].all():
    #         print(i, "true")
