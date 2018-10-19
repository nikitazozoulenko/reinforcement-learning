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

from network import FCC2x2
from cube2x2 import Cube, cube_to_tensor
from mcts import MCTS, eps_greedy
from graphing import Grapher


device = torch.device("cuda")
model_path="save_dir/model.pth"
results_path = "save_dir/results2x2.pkl"

def create_agent():
    agent_network = FCC2x2().to(device)
    agent_network.load_state_dict(torch.load(model_path))
    agent_network.eval()
    agent_mcts = MCTS(agent_network)
    return agent_mcts


def single_mcts(agent_mcts, eps):
    agent_mcts.root.uct_traverse(eps)
    return agent_mcts.root.solution_found


def attempt_random_cube(agent_mcts, eps, n_shuffle):
    #solve cube
    solve_time = time.time()
    agent_mcts.reset(n_shuffle)
    for i in range(100000000):
        solution_found = single_mcts(agent_mcts, eps)
        if solution_found:
            break
    solve_time = time.time() - solve_time
    return solution_found, solve_time, i


def eval(agent_mcts, eps, n_shuffle, n_eval):
    results = []
    for i in range(n_eval):
        solution_found, solve_time, mcts_search_steps = attempt_random_cube(agent_mcts, eps, n_shuffle)
        results += [[solution_found, solve_time, mcts_search_steps]]
        print(solution_found, solve_time, mcts_search_steps)
    with open(results_path, "wb") as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


def read():
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    results = np.array(results).reshape(-1, 3)
    print("average solve rate", np.mean(results[:, 0]))
    print("average solve speed", np.mean(results[:, 1]))
    print("average mcts steps needed to find solution", np.mean(results[:, 2]))

    
if __name__ == "__main__":
    #mcts variables
    eps=0

    #eval variables
    n_eval = 1000
    n_shuffle = 500
    agent_mcts = create_agent()

    if os.path.exists(results_path):
        print("results file already existed")
        read()
    else:
        eval(agent_mcts, eps, n_shuffle, n_eval)
        read()