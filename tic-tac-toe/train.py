import random
import collections

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from network import FCC
from environment import GameBoard, step
from graphing import graph, values2ewma
from mcts import MCTS, eps_greedy, board_to_tensor

device = torch.device("cuda")

class ReplayMemory():
    def __init__(self, maxlen):
        self.deque = collections.deque(maxlen=maxlen)
        self.maxlen = maxlen
    

    def add(self, sars):
        if len(self.deque) >= self.maxlen:
            self.deque.popleft()
        self.deque.append(sars)

    
    def sample(self, batch_size=64):
        if len(self.deque) < batch_size:
            size = len(self.deque)
        else:
            size = batch_size
        samples = random.sample(self.deque, size)
        return samples


def test():
    board_size = 3
    win_length = 3

    opponent_network = FCC(board_size).to(device)
    opponent_mcts = MCTS(board_size, win_length, opponent_network)
    agent_network = FCC(board_size).to(device)
    agent_mcts = MCTS(board_size, win_length, agent_network)

    players = [opponent_mcts, agent_mcts]
    
    game_board = GameBoard(board_size, win_length)

if __name__ == "__main__":
    test()


