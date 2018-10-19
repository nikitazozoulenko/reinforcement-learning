import random
import collections
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import FCC2x2
from cube2x2 import Cube, cube_to_tensor
from mcts import MCTS, eps_greedy
from graphing import Grapher


device = torch.device("cuda")
model_path="save_dir/model.pth"
experience_path="save_dir/exp_replay.pkl"

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


class Player:
    def __init__(self, mcts, experience_replay):
        self.mcts = mcts
        self.experience_replay = experience_replay
    

    def monte_carlo_tree_search(self, max_mcts_steps, mcts_eps, final_choose_eps):
        return self.mcts.monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps)


    def change_root_with_action(self, action):
        self.mcts.change_root_with_action(action)


    def add_to_experience_replay(self, node):
        states = cube_to_tensor(node.s)
        tree_Q = node.tree_Q.data
        for state in states:
            self.experience_replay.add([state, tree_Q])


class OptimizerHandler:
    def __init__(self, agent, batch_size, learning_rate, n_train_per_solve, n_eval, min_solve_rate, n_shuffle):
        self.agent=agent
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_train_per_solve = n_train_per_solve
        self.n_eval = n_eval
        self.min_solve_rate = min_solve_rate
        self.n_shuffle=n_shuffle

        self.create_optim()
        self.create_grapher()
        self.create_results_tracker()

        self.MSE = nn.MSELoss()


    def create_results_tracker(self):
        self.n_latest_wins = 0
        self.n_latest_losses = 0
        self.deque_latest_results = collections.deque(maxlen=self.n_eval)


    def create_grapher(self):
        self.grapher = Grapher("save_dir/loss_folder/losses_n"+ str(self.n_shuffle)+".txt")
        

    def create_optim(self):
        self.optimizer = optim.SGD(self.agent.mcts.network.parameters(), weight_decay=0.0001, lr=self.learning_rate, momentum=0.9)


    def optimize_model(self):
        model = self.agent.mcts.network
        experience_replay = self.agent.experience_replay
        model.train()
        if len(experience_replay.deque) > self.batch_size:
            samples = experience_replay.sample(self.batch_size)
            s, target = zip(*samples)
            s = torch.stack(s)
            target = torch.stack(target)
            self.optimizer.zero_grad()
            loss = self.MSE(model(s), target)
            loss.backward()
            self.optimizer.step()

            self.grapher.write(str(loss.data.cpu().numpy()))
        model.eval()

    
    def save_model_and_reset_grapher(self):
        agent_network = self.agent.mcts.network
        torch.save(agent_network.state_dict(), model_path)
        with open(experience_path, "wb") as f:
            pickle.dump(self.agent.experience_replay, f, pickle.HIGHEST_PROTOCOL)
        self.create_grapher()


    def train(self, max_mcts_steps, mcts_eps, final_choose_eps):
        for i in range(10000000):
            self.check_if_increase_n_shuffle(i)
            self.attempt_random_cube(max_mcts_steps, mcts_eps, final_choose_eps)
            for _ in range(self.n_train_per_solve):
                self.optimize_model()
            

    def check_if_increase_n_shuffle(self, i):
        if len(self.deque_latest_results) >= self.n_eval:
            solve_rate = self.n_latest_wins/self.n_eval
            if i % 10==0:
                print(i, "solve_rate", solve_rate)
            if solve_rate > self.min_solve_rate:
                self.n_shuffle+=1
                self.create_results_tracker()
                self.save_model_and_reset_grapher()
                print("solve rate reached", solve_rate)
                print("saving model and increasing nshuffle to", self.n_shuffle)


    def attempt_random_cube(self, max_mcts_steps, mcts_eps, final_choose_eps):
        #solve cube
        self.agent.mcts.reset(self.n_shuffle)
        for _ in range(min(50, self.n_shuffle*2)):
            a, terminate = self.agent.mcts.monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps)
            if terminate:
                break
        self.traverse_and_add_to_replay()

        #update eval statistics
        if len(self.deque_latest_results) == self.n_eval:
            first = self.deque_latest_results.popleft()
            if first == True:
                self.n_latest_wins -= 1
            else:
                self.n_latest_losses -= 1
        
        self.deque_latest_results.append(terminate)
        if terminate == True:
            self.n_latest_wins += 1
        else:
            self.n_latest_losses += 1


    def traverse_and_add_to_replay(self):
        node = self.agent.mcts.root.parent
        while node.parent != None:
            node = node.parent
            self.agent.add_to_experience_replay(node)
        self.agent.add_to_experience_replay(node)


def create_agent(replay_maxlen):
    if not os.path.exists(model_path):
        torch.save(FCC2x2().to(device).state_dict(), model_path)
    agent_network = FCC2x2().to(device)
    agent_network.load_state_dict(torch.load(model_path))
    agent_network.eval()
    agent_mcts = MCTS(agent_network)
    if os.path.exists(experience_path):
        with open(experience_path, "rb") as f:
            exp_replay = pickle.load(f)
    else:
        exp_replay = ExperienceReplay(replay_maxlen)
    agent = Player(agent_mcts, exp_replay)
    return agent


def main():
    #mcts variables
    max_mcts_steps=800
    mcts_eps=0.05
    final_choose_eps=0

    #train variables
    replay_maxlen = 10000
    batch_size =1024
    learning_rate = 0.0001
    n_train_per_solve = 5
    n_eval = 1000
    min_solve_rate = 0.9
    n_shuffle = 307
    
    #optimizer handler
    optimizer_handler = OptimizerHandler(create_agent(replay_maxlen), batch_size, learning_rate, n_train_per_solve, n_eval, min_solve_rate, n_shuffle)

    #train on some cubes
    optimizer_handler.train(max_mcts_steps, mcts_eps, final_choose_eps)


if __name__ == "__main__":
    main()