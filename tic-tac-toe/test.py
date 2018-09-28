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
from graphing import graph, values2ewma
from mcts import MCTS, eps_greedy, board_to_tensor


def test():
    game_board = GameBoard(size = 3, win_length=3)

    game_board.take_action(3)
    game_board.reverse_player_positions()
    print(game_board.get_allowed_actions())


if __name__ == "__main__":
    test()