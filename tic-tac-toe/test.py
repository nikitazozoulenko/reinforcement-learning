import random
import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import FCC
from environment import GameBoard, step
from mcts import MCTS, eps_greedy, board_to_tensor


def test():
    size = 7

    game_board = GameBoard(size=size,win_length=3)
    board = game_board.board
    win_length=game_board.win_length


    coords = []
    count = 0

    index = 6

    for i in range(size-abs(index-size+1)):
        if index <= size-1:
            coord = (i, size-1-abs(index-size+1)-i)
        else: #if index > self.size-1
            coord = (i+abs(index-size+1), size-1-i)
        print(coord, board[coord])
        if board[coord] == 1:
            count += 1
            coords += [coord]
        if count == win_length:
            return coords
        else:
            count = 0
            coords = []

if __name__ == "__main__":
    test()