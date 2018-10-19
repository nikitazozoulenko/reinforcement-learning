import random
import collections
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import CNN
from environment import GameBoard, step
from mcts import MCTS, eps_greedy
from graphing import Grapher


device = torch.device("cuda")
model_path="save_dir/model.pth"
experience_path="save_dir/exp_replay.pkl"
losses_path="save_dir/loss_folder/losses.txt"

class ExperienceReplay:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.reset()
    

    def add(self, sars):
        if len(self.deque) >= self.maxlen:
            self.deque.popleft()
        self.deque.append(sars)

    
    def sample(self, batch_size):
        if len(self.deque) < batch_size:
            size = len(self.deque)
        else:
            size = batch_size
        samples = random.sample(self.deque, size)
        return samples

    
    def reset(self):
        self.deque = collections.deque(maxlen=self.maxlen)


class MatchHandler:
    def __init__(self, agent_mcts, opponent_mcts, experience_replay, game_board, n_eval):
        self.agent_mcts = agent_mcts
        self.opponent_mcts = opponent_mcts
        self.experience_replay = experience_replay
        self.game_board = game_board
        self.n_eval = n_eval

        self.create_results_tracker()


    def play_turn(self, player1, player2, mcts_steps, eps):
        '''Plays a turn. self.game_board gets stepped once, both player mcts trees change roots. 
           Returns float reward and bool terminate'''
        a = player1.monte_carlo_tree_search(mcts_steps, eps)
        r, terminate = self.game_board.step(a)
        player1.change_root_with_action(a)
        player2.change_root_with_action(a)
        return r, terminate

    
    def play_match(self, mcts_steps, eps):
        '''Plays a single match, tallies the result, and adds to experience replay'''
        self.game_board.reset()
        self.agent_mcts.reset()
        self.opponent_mcts.reset()

        #who starts first?
        players = [self.opponent_mcts, self.agent_mcts]
        agent_starts = bool(random.randint(0,1))
        if agent_starts:
            players.reverse()

        #play match
        terminate = False
        while not terminate:
            r, terminate = self.play_turn(*players, mcts_steps, eps)
            players.reverse()

        #experience replay
        self.agent_mcts.traverse_and_add_to_replay(self.experience_replay, torch.FloatTensor([r]).to(device))

        #update eval statistics
        self.update_eval_statistics(agent_starts, r)


    def update_eval_statistics(self, agent_starts, r):
        '''Takes in (bool) agent_wins, and (float) r (last reward before game terminates), 
           and updates last self.n_eval game stats'''
        if len(self.deque_latest_results) == self.n_eval:
            first = self.deque_latest_results.popleft()
            if first == 1:
                self.n_latest_wins -= 1
            elif first == 0:
                self.n_latest_draws -= 1
            else:
                self.n_latest_losses -= 1
        if not agent_starts:
            r = -r
        self.deque_latest_results.append(r)
        if r == 0:
            self.n_latest_draws += 1
        elif r == 1:
            self.n_latest_wins += 1
        elif r == -1:
            self.n_latest_losses += 1
    

    def create_results_tracker(self):
        '''Reset tally and deque with last (n_eval) games'''
        self.n_latest_wins = 0
        self.n_latest_losses = 0
        self.n_latest_draws = 0
        self.deque_latest_results = collections.deque(maxlen=self.n_eval)


class OptimizerHandler:
    def __init__(self, match_handler, batch_size, learning_rate, n_train_per_game, min_win_rate):
        self.match_handler = match_handler
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_train_per_game = n_train_per_game
        self.min_win_rate = min_win_rate

        self.create_grapher()
        self.create_optim()
        self.MSE = nn.MSELoss()
    

    def create_grapher(self):
        self.grapher = Grapher(losses_path)
        

    def create_optim(self):
        self.optimizer = optim.SGD(self.match_handler.agent_mcts.network.parameters(), weight_decay=0.001, lr=self.learning_rate, momentum=0.9)
        self.eps = torch.FloatTensor([1e-8]).to(device)
    

    def optimize_model(self):
        '''Applies one iteration of SGD on agent network (if batch size is sufficiently large),
           target is sampled from experience replay'''
        model = self.match_handler.agent_mcts.network
        experience_replay = self.match_handler.experience_replay
        model.train()
        if len(experience_replay.deque) > self.batch_size:
            samples = experience_replay.sample(self.batch_size)
            gt_probs, gt_v, s = zip(*samples)
            gt_probs = torch.stack(gt_probs)
            gt_v = torch.stack(gt_v)
            s = torch.stack(s)

            self.optimizer.zero_grad()
            probs, v = model(s)
            loss = self.MSE(v, gt_v) - torch.mean(gt_probs*torch.log(probs+self.eps))
            loss.backward()
            self.optimizer.step()

            self.grapher.write(str(loss.data.cpu().numpy()))
        model.eval()
    

    def train(self, mcts_steps, eps):
        '''Training loop, one self-play match and self.n_train_per_game SGD iterations'''
        for i in range(10000000):
            self.replace_opponent_if_needed(i)
            self.match_handler.play_match(mcts_steps, eps)
            for _ in range(self.n_train_per_game):
                self.optimize_model()
    

    def replace_opponent_if_needed(self, i):
        '''Checks if the current winrate is above a certain threshold.
           If yes, updates opponent and saves agent'''
        n_eval = self.match_handler.n_eval
        n_wins = self.match_handler.n_latest_wins
        n_losses = self.match_handler.n_latest_losses
        n_draws = self.match_handler.n_latest_draws
        if len(self.match_handler.deque_latest_results) >= n_eval:
            win_rate = n_wins/(n_wins+n_losses)
            if i % 10==0:
                print("iter", i, "win_rate:", win_rate, "wins:", n_wins, "losses:", n_losses, "draws:", n_draws)
            if win_rate > self.min_win_rate:
                self.save_model_and_update_opponent()
                self.match_handler.create_results_tracker()
                print("\nwin rate reached", win_rate, "saving model\n")
    

    def save_model_and_update_opponent(self):
        '''Saves agent network to file and loads it into opponent'''
        agent_network = self.match_handler.agent_mcts.network
        opponent_network = self.match_handler.opponent_mcts.network
        torch.save(agent_network.state_dict(), model_path)
        opponent_network.load_state_dict(torch.load(model_path))


def create_agent_and_opponent(board_size, win_length, replay_maxlen):
    #network and exp replay
    if not os.path.exists(model_path):
        torch.save(CNN(board_size).to(device).state_dict(), model_path)
    if os.path.exists(experience_path):
        with open(experience_path, "rb") as f:
            exp_replay = pickle.load(f)
    else:
        exp_replay = ExperienceReplay(replay_maxlen)

    #agent
    agent_network = CNN(board_size).to(device)
    agent_network.load_state_dict(torch.load(model_path))
    agent_network.eval()
    agent_mcts = MCTS(board_size, win_length, agent_network)

    #opponent
    opponent_network = CNN(board_size).to(device)
    opponent_network.load_state_dict(torch.load(model_path))
    opponent_network.eval()
    opponent_mcts = MCTS(board_size, win_length, opponent_network)

    return agent_mcts, opponent_mcts, exp_replay


def main():
    #game variables
    board_size = 8
    win_length = 5

    #mcts variables
    mcts_steps=300
    eps=0.05

    #train variables
    replay_maxlen = 10000
    batch_size = 1024
    learning_rate = 0.001
    n_train_per_game = 5
    n_eval = 500
    min_win_rate = 0.55

    #training things
    match_handler = MatchHandler(*create_agent_and_opponent(board_size, win_length, replay_maxlen), GameBoard(board_size, win_length), n_eval)
    optimizer_handler = OptimizerHandler(match_handler, batch_size, learning_rate, n_train_per_game, min_win_rate)
    optimizer_handler.train(mcts_steps, eps)


if __name__ == "__main__":
    main()