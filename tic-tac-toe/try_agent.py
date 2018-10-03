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
from train import model_path, create_agent_and_opponent, MatchHandler

def main():
    #variables
    board_size = 7
    win_length = 4
    max_mcts_steps=1
    mcts_eps=0.0
    final_choose_eps=0
    replay_maxlen = 5000

    #match handler
    match_handler = MatchHandler(*create_agent_and_opponent(board_size, win_length, replay_maxlen))

    #play some games
    match_handler.play_match(max_mcts_steps, mcts_eps, final_choose_eps, do_print=True)


if __name__ == "__main__":
    main()