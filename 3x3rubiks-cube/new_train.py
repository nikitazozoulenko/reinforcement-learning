import random
import collections
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import FCC
from cube import Cube, cube_to_tensor
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
    def __init__(self, agent, batch_size, learning_rate, n_train_per_solve, n_eval, min_mcts_steps, n_shuffle):
        self.agent=agent
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_train_per_solve = n_train_per_solve
        self.n_eval = n_eval
        self.min_mcts_steps = min_mcts_steps
        self.n_shuffle=n_shuffle

        self.create_optim()
        self.create_grapher()
        self.create_results_tracker()

        self.MSE = nn.MSELoss()


    def create_results_tracker(self):
        self.average_mcts_steps = -1
        self.average_solve_time = -1
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


    def train(self, eps):
        for i in range(10000000):
            self.check_if_increase_n_shuffle(i)
            self.attempt_random_cube(eps)
            for _ in range(self.n_train_per_solve):
                self.optimize_model()
            

    def check_if_increase_n_shuffle(self, i):
        if i % 10==0:
            print("it", i, "avg steps", self.average_mcts_steps, "avg time", self.average_solve_time, "n_eval", len(self.deque_latest_results))
        if len(self.deque_latest_results) >= self.n_eval:
            if self.average_mcts_steps < self.min_mcts_steps:
                self.n_shuffle+=1
                print("\n\n", "mcts length reached", self.average_mcts_steps)
                print("saving model and increasing nshuffle to", self.n_shuffle, "\n\n")
                self.create_results_tracker()
                self.save_model_and_reset_grapher()
 


    def attempt_random_cube(self, eps):
        #solve cube
        self.agent.mcts.reset(self.n_shuffle)
        solve_time = time.time()
        for mcts_steps in range(10000):
            a = self.agent.mcts.monte_carlo_tree_search(1, eps)
            terminate = self.agent.mcts.root.solution_found
            if terminate:
                break
            if mcts_steps==9999:
                print("reached max search steps")
        terminate = self.agent.mcts.root.is_terminate_state
        for _ in range(min(40, int(self.n_shuffle*1.5))):
            a = self.agent.mcts.get_best_action(eps=0)
            terminate = self.agent.mcts.change_root_with_action(a)
            if terminate:
                break
        solve_time = time.time()-solve_time
        self.traverse_and_add_to_replay()

        #update eval statistics
        n = len(self.deque_latest_results)
        if n == self.n_eval:
            first_mcts_steps, first_time = self.deque_latest_results.popleft()
            self.average_mcts_steps = (n*self.average_mcts_steps - first_mcts_steps)/(n-1)
            self.average_solve_time = (n*self.average_solve_time - first_time)/(n-1)
        self.deque_latest_results.append([mcts_steps, solve_time])
        self.average_mcts_steps = (n*self.average_mcts_steps + mcts_steps)/(n+1)
        self.average_solve_time = (n*self.average_solve_time + solve_time)/(n+1)


    def traverse_and_add_to_replay(self):
        node = self.agent.mcts.root.parent
        while node.parent != None:
            node = node.parent
            self.agent.add_to_experience_replay(node)
        self.agent.add_to_experience_replay(node)


def create_agent(replay_maxlen):
    if not os.path.exists(model_path):
        torch.save(FCC().to(device).state_dict(), model_path)
    agent_network = FCC().to(device)
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
    eps=0.05

    #train variables
    replay_maxlen = 10000
    batch_size =1024
    learning_rate = 0.003
    n_train_per_solve = 5
    n_eval = 1000
    min_average_mcts_steps = 1200
    n_shuffle = 8
    
    #optimizer handler
    optimizer_handler = OptimizerHandler(create_agent(replay_maxlen), batch_size, learning_rate, n_train_per_solve, n_eval, min_average_mcts_steps, n_shuffle)

    #train on some cubes
    optimizer_handler.train(eps)


if __name__ == "__main__":
    main()