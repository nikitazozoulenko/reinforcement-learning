import time

import numpy as np
from scipy import signal

default_starve = 2
default_overpop = 3
default_birth = 3
default_stride = 1
default_game_size = 21

def read_coords_from_file(filename="coords.txt"):
    '''Reads the file filename and returns a list of [y, x] coordinates.'''
    coords = []
    with open(filename, "r") as f:
        for line in f.readlines():
            coord = line.strip().split(" ")
            coord = [int(n) for n in coord]
            coords += [coord]
    return coords


def assert_number_or_default(usr_input, default):
    '''Checks if the user input is a valid number with error handling and returns the correct int.'''
    try:
        if usr_input == "":
            number = default
        else:
            number = int(usr_input)
    except Exception:
        print("Did not insert valid number, using default =", default)
        number = default
    return number


def read_rules_from_input():
    '''Reads game world rules from standard input and returns the starve limit, overpopulation limit and frame stride.'''
    default_starve = 2
    default_overpop = 3
    default_birth = 3
    default_stride = 1
    starve = input("A cell starves to death below this many neighbours (Default = "+str(default_starve)+"): ")
    starve = assert_number_or_default(starve, default_starve)
    overpop = input("A cell dies by overpopulation above this many neighbours (Default = "+str(default_overpop)+"): ")
    overpop = assert_number_or_default(overpop, default_overpop)
    birth = input("A cell is birthed if it has this many neighbours (Default = "+str(default_birth)+"): ")
    birth = assert_number_or_default(birth, default_birth)
    stride = input("Generation skip stride (Default = "+str(default_stride)+"): ")
    stride = assert_number_or_default(stride, default_stride)
    return starve, overpop, birth, stride


class GameWorld:
    '''The GameWorld encorporates every concept from game logic, to game rule 
       creation, to coordinate loading'''
    def __init__(self, starve, overpop, birth, stride, size=default_game_size):
        self.size = size
        self.world = np.zeros((size, size), dtype=np.bool)
        self.kernel = np.ones((3,3))
        self.update_rules(starve, overpop, birth, stride, size)


    def set_from_coords(self, coords):
        '''Takes in a list of cell coordinates and adds them to the game world'''
        for [x, y] in coords:
            if -1<x and x<self.size:
                if -1<y and y<self.size:
                    self.world[x,y] = True


    def step(self):
        '''Steps through and changes the game world self.stide iterations'''
        for _ in range(self.stride):
            n_neighbours = signal.convolve2d(self.world, self.kernel, mode="same") - self.world
            mask_birth = n_neighbours == self.birth
            mask_overpop = n_neighbours < self.overpop+1
            mask_starve = n_neighbours > self.starve-1
            self.world = (mask_birth + self.world*(mask_overpop*mask_starve))>0


    def reset(self):
        '''Resets game world to no livivng cells'''
        self.world = np.zeros((self.size, self.size), dtype=np.bool)

    
    def update_rules(self, starve, overpop, birth, stride, new_size):
        '''Takes in the starve, overpopulation, birthing, stride, and game board size rules
           and updates the game logic'''
        self.starve = starve
        self.overpop = overpop
        self.birth = birth
        self.stride = stride
        
        new_world = np.zeros((new_size, new_size), dtype=np.bool)
        if new_size<self.size:
            new_world[:, :] = self.world[:new_size, :new_size]
        else:
            new_world[:self.size, :self.size] = self.world[:, :]
        self.world = new_world
        self.size = new_size


    def __str__(self):
        '''Converts the game world into a visually appealing string'''
        string = ""
        for row in self.world:
            for ele in row:
                val = "0" if ele else "-"
                string += val
            string += "\n"
        return string


def main():
    size = 12
    coords = read_coords_from_file()
    rules = read_rules_from_input()
    world = GameWorld(*rules, size)
    world.set_from_coords(coords)
    for _ in range(100):
        print(world)
        world.step()
        time.sleep(0.3)


if __name__ == "__main__":
    main()