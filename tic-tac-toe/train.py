import random
import collections
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from network import FCC
from environment import GameBoard, step
from graphing import graph, values2ewma
from mcts import MCTS, eps_greedy, board_to_tensor

device = torch.device("cuda")

class ExperienceReplay:
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


class Player:
    def __init__(self, mcts, experience_replay):
        self.mcts = mcts
        self.experience_replay = experience_replay
    

    def monte_carlo_tree_search(self, max_mcts_steps, mcts_eps, final_choose_eps):
        return self.mcts.monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps)


    def change_root_with_action(self, action):
        self.mcts.change_root_with_action(action)

    
    def add_to_experience_replay(self):
        s = board_to_tensor(self.mcts.root.s)[0]
        tree_Q = self.mcts.root.tree_Q.data
        self.experience_replay.add([s, tree_Q])


class MatchHandler:
    def __init__(self, agent, opponent, game_board):
        self.agent = agent
        self.opponent = opponent
        self.game_board = game_board
        self.reset_tally_results()

    
    def play_match(self, max_mcts_steps, mcts_eps, final_choose_eps):
        #randomize player start
        self.game_board.reset()
        self.agent.mcts.reset()
        self.opponent.mcts.reset()
        players = [self.agent, self.opponent]
        agent_starts = bool(random.randint(0,1))
        if not agent_starts:
            players.reverse()

        #turn loop until terminate
        _ = players[1].monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps) #player waiting gets to start thinking first
        while True:
            end_match, coords = self.play_turn(players[0], players[1], max_mcts_steps, mcts_eps, final_choose_eps)
            if end_match:
                break
            players.reverse()

        #evaluate who won and tally results
        if coords:
            board_value = int(self.game_board.board[coords[0]])
            is_cross = True if board_value==1 else False
            if is_cross == agent_starts:
                self.wins_agent += 1
            else:
                self.losses_agent += 1
        else:
            self.draws_agent += 1


    def play_turn(self, player, opponent, max_mcts_steps, mcts_eps, final_choose_eps):
        picked_action = player.monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps)
        self.game_board.take_action(picked_action)

        for person in [player, opponent]:
            if person.experience_replay != None:
                person.add_to_experience_replay()
            person.change_root_with_action(picked_action)

        terminate, coords = self.game_board.check_win_position()
        # print("game board")
        # print(terminate, coords)
        # print(self.game_board)
        # print()
        # print()
        return terminate, coords

    
    def reset_tally_results(self):
        self.wins_agent = 0
        self.draws_agent = 0
        self.losses_agent = 0
    

class OptimizerHandler:
    def __init__(self, match_handler, learning_rate=0.001):
        self.match_handler=match_handler
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.match_handler.agent.mcts.network.parameters(), lr=learning_rate)

        self.MSE = nn.MSELoss()


    def optimize_model(self, n_train=1, batch_size=64):
        model = self.match_handler.agent.mcts.network
        experience_replay = self.match_handler.agent.experience_replay
        for i in range(n_train):
            samples = experience_replay.sample()
            s, target = zip(*samples)
            s = torch.stack(s)
            target = torch.stack(target)

            loss = self.MSE(model(s), target)
            #TODO TODO TODO DO LEARNING HERE
            #TODO TODO TODO
            #TODO TODO TODO
            print("START")
            print(s)
            print(s.size())
            print(target)
            print(target.size())
            print("END")
            assert True==False

    
    def update_opponent_if_needed(self, min_n_games=100):
        n_games = self.match_handler.wins_agent + self.match_handler.draws_agent + self.match_handler.losses_agent
        if n_games > min_n_games:
            pass
            #TODO DO STUFF
            #TODO save agent
            #TODO then replace opponents state dict with agent
            #TODO maybe 


def create_agent_and_opponent(board_size, win_length, replay_maxlen, path="save_dir/model.pth"):
    if not os.path.exists(path):
        torch.save(FCC(board_size).to(device).state_dict(), path)

    #opponent
    opponent_network = FCC(board_size).to(device)
    opponent_network.load_state_dict(torch.load(path))
    opponent_network.eval()
    opponent_mcts = MCTS(board_size, win_length, opponent_network)
    opponent = Player(opponent_mcts, None)

    #agent
    agent_network = FCC(board_size).to(device)
    agent_network.load_state_dict(torch.load(path))
    agent_network.eval()
    agent_mcts = MCTS(board_size, win_length, agent_network)
    agent = Player(agent_mcts, ExperienceReplay(replay_maxlen))
    return agent, opponent, GameBoard(board_size, win_length)





def main():
    #variables
    board_size = 3
    win_length = 3
    max_mcts_steps=100
    mcts_eps=0.1
    final_choose_eps=0
    replay_maxlen = 1000

    #match handler
    match_handler = MatchHandler(*create_agent_and_opponent(board_size, win_length, replay_maxlen))
    optimizer_handler = OptimizerHandler(match_handler)

    #play some games
    for i in range(10000):
        match_handler.play_match(max_mcts_steps, mcts_eps, final_choose_eps)
        print("wins", match_handler.wins_agent)
        print("draws", match_handler.draws_agent)
        print("losses", match_handler.losses_agent)
        optimizer_handler.optimize_model()
        optimizer_handler.update_opponent_if_needed()


if __name__ == "__main__":
    main()


