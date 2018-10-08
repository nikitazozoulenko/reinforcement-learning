import random
import collections
import os

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


class MatchHandler:
    def __init__(self, agent):
        self.agent = agent
        self.reset_tally_results()


    def solve_one_cube(self, max_mcts_steps, mcts_eps, final_choose_eps, n_shuffle):
        #play one cube,  MAX TURNS == 2*N_SHUFFLE
        self.agent.mcts.reset(n_shuffle)
        for _ in range(2*n_shuffle):
            _a, terminate = self.agent.mcts.monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps)
            if terminate:
                break

        #tally results
        if terminate:
            self.n_did_solve += 1
        else:
            self.n_didnt_solve += 1

        #traverse tree backwards and add to experience replay
        self.traverse_and_add_to_replay()


    def traverse_and_add_to_replay(self):
        node = self.agent.mcts.root.parent
        while node.parent != None:
            node = node.parent
            self.agent.add_to_experience_replay(node)
        self.agent.add_to_experience_replay(node)


    def reset_tally_results(self):
        self.n_did_solve = 0
        self.n_didnt_solve = 0


    ##### This traverse method didnt work to learn the network how to do stuff right
    # def traverse_and_add_to_replay(self):
    #     '''Traverses to root, then recursively adds ALL visited states to replay memory'''
    #     root = self.agent.mcts.get_original_root()
    #     self._recursive_traverse_add(root)


    # def _recursive_traverse_add(self, node):
    #     self.agent.add_to_experience_replay(node)
    #     for child in node.children:
    #         if child != None:
    #             if not child.is_terminate_state:
    #                 self._recursive_traverse_add(child)



    

class OptimizerHandler:
    def __init__(self, match_handler, batch_size, n_iter_train, learning_rate):
        self.match_handler=match_handler
        self.batch_size = batch_size
        self.n_iter_train = n_iter_train
        self.learning_rate = learning_rate
        self.MSE = nn.MSELoss()

        self.n_shuffle=3

        self.create_optim()
        self.create_grapher()

    
    def create_grapher(self):
        self.grapher = Grapher("save_dir/loss_folder/losses_n"+ str(self.n_shuffle)+".txt")
        

    def create_optim(self):
        self.optimizer = optim.SGD(self.match_handler.agent.mcts.network.parameters(), weight_decay=0.0001, lr=self.learning_rate, momentum=0.9)
        # self.optimizer = optim.Adam(self.match_handler.agent.mcts.network.parameters(), weight_decay=0.0001, lr=self.learning_rate)


    def optimize_model(self):
        model = self.match_handler.agent.mcts.network
        experience_replay = self.match_handler.agent.experience_replay
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
    

    def save_model(self):
        agent_network = self.match_handler.agent.mcts.network
        torch.save(agent_network.state_dict(), model_path)


    def save_model_and_reset_optim_and_grapher(self):
        self.save_model()
        self.match_handler.reset_tally_results()
        self.create_optim()
        self.create_grapher()
        self.match_handler.agent.experience_replay.reset()

    
    def train(self, max_mcts_steps, mcts_eps, final_choose_eps, n_eval):
        while True:
            #eval
            solve_rate = self.evaluate_agent(max_mcts_steps, mcts_eps, final_choose_eps, n_eval)
            print("solve rate", solve_rate, "n_shuffle", self.n_shuffle)
            if solve_rate > 0.9:
                self.n_shuffle+=1
                self.match_handler.reset_tally_results()
                self.save_model_and_reset_optim_and_grapher()
                print("saving model and increasing nshuffle to", self.n_shuffle)
                continue

            #train
            # for _ in range(self.n_iter_solve_cube):
            #     self.match_handler.solve_one_cube(max_mcts_steps, mcts_eps, final_choose_eps, self.n_shuffle)
            for _ in range(self.n_iter_train):
                self.optimize_model()



    def evaluate_agent(self, max_mcts_steps, mcts_eps, final_choose_eps, n_eval):
        '''Evaluates the agent (n_eval) times where the agent is allowed max (n_shuffle*2) turns on cube shuffled (n_shuffle) times'''
        n_times_solved = 0
        for _ in range(n_eval):
            self.match_handler.agent.mcts.reset(self.n_shuffle)
            for _ in range(self.n_shuffle*2):
                a, terminate = self.match_handler.agent.mcts.monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps)
                if terminate:
                    n_times_solved+=1
                    break
            self.match_handler.traverse_and_add_to_replay()
        return n_times_solved/n_eval



def create_agent(replay_maxlen):
    if not os.path.exists(model_path):
        torch.save(FCC2x2().to(device).state_dict(), model_path)
    agent_network = FCC2x2().to(device)
    agent_network.load_state_dict(torch.load(model_path))
    agent_network.eval()
    agent_mcts = MCTS(agent_network)
    agent = Player(agent_mcts, ExperienceReplay(replay_maxlen))
    return agent



def main():
    #variables
    max_mcts_steps=100
    mcts_eps=0.05
    final_choose_eps=0
    replay_maxlen = 1000000
    batch_size = 8092
    n_iter_train = 1000
    learning_rate = 0.01
    n_eval = 100
    
    #optimizer handler
    match_handler = MatchHandler(create_agent(replay_maxlen))
    optimizer_handler = OptimizerHandler(match_handler, batch_size, n_iter_train, learning_rate)

    #train on some cubes
    optimizer_handler.train(max_mcts_steps, mcts_eps, final_choose_eps, n_eval)


if __name__ == "__main__":
    main()