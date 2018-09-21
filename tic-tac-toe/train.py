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


class Player():
    def __init__(self, mcts, replay_memory, person):
        self.mcts = mcts
        self.replay_memory = replay_memory
        self.person = person #agent, opponent


def main():
    board_size = 3
    win_length = 3
    max_mcts_steps=1000
    mcts_eps=1
    final_choose_eps=0

    opponent_network = FCC(board_size).to(device)
    opponent_network.eval()
    opponent_mcts = MCTS(board_size, win_length, opponent_network)
    agent_network = FCC(board_size).to(device)
    agent_network.eval()
    agent_mcts = MCTS(board_size, win_length, agent_network)
    game_board = GameBoard(board_size, win_length)

    play_match(game_board, agent_mcts, opponent_mcts, max_mcts_steps, mcts_eps, final_choose_eps)


def play_match(game_board, player1, player2, max_mcts_steps, mcts_eps, final_choose_eps):
    players = [player1, player2]
    if random.randint(0,1):
        players.reverse()
    while True:
        terminate, _ = play_turn(game_board, players[0], players[1], max_mcts_steps, mcts_eps, final_choose_eps)
        if terminate:
            break
        players.reverse()
        

def play_turn(game_board, player, opponent, max_mcts_steps, mcts_eps, final_choose_eps):
    action = player.monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps)
    game_board.take_action(action)
    player.change_root_with_action(action)
    opponent.change_root_with_action(action)
    terminate, coords = game_board.check_win_position()
    print("game board")
    print(terminate, coords)
    print(game_board)
    print()
    print()
    return terminate, coords


if __name__ == "__main__":
    main()


