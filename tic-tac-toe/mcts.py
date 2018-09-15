import random
import collections

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from network import FCC2x2
from cube2x2 import Cube
from graphing import graph, values2ewma

device = torch.device("cuda")
QNet = FCC2x2().to(device)
QNet.eval()


def cube_to_tensor(s):
    tensor = torch.from_numpy(s.cube_array)
    return tensor.to(device).view(1, -1)


def eps_greedy(action_values, eps):
    if eps == 0 or np.random.rand() > eps:
        q, a = torch.max(action_values, dim=-1)
    else:
        a = np.random.randint(action_values.size(-1))
        a = torch.tensor([a], dtype=torch.long, device=device)
        q = result[0,a]
    return q, a


def step(s, a):
    s_prime = s.copy().take_action(a)
    terminate = s_prime.check_if_solved()
    if terminate:
        r = 40
    else:
        r = -1
    return s_prime, r, terminate


class MCTS():
    def __init__(self, root_state):
        self.root = Node(s=root_state, prev_a=None, parent=None, depth=0, terminate=False)
        self.t = 0


    def monte_carlo_tree_search(self):
        while self.t<1:
            leaf = self.root.uct_traverse()
            leaf.simulate(eps=0.00)
            self.t +=1
        return self.best_path()


    def best_path(self):
        pass
    

class Node():
    def __init__(self, s, prev_a, parent, depth, terminate):
        self.parent = parent
        self.s = s
        self.prev_a=prev_a
        self.depth= depth
        self.terminate=terminate
        self.Q = QNet(cube_to_tensor(s))
        self.tree_Q = np.zeros(self.Q.size(-1)) #torch.zeros((1, self.Q.size(-1)), device=device)
        self.children = [None] * self.Q.size(-1)
        self.visited = 1


    def uct_traverse(self):
        children_n_visit = []
        for child in self.children:
            if child == None:
                return self
            children_n_visit += [child.visited]
        children_n_visit = torch.tensor(visited, device=device)

        U = (np.log(self.visited) / children_n_visit) ** 0.5
        q, a = eps_greedy(self.Q+U, eps=0.01)
        return self.children[a].uct_traverse()


    def simulate(self, eps):
        #########################TODO TODO TODO eps not implemented yet TODO TODO TODO############################
        q, a = torch.sort(Q, dim=-1, descending=True)
        for action in a[0]:
            if self.children[action] != None:
                a = action
                break
        
        s_prime, r, terminate = step(self.s, a)
        self.children[action] = Node(s=s_prime, prev_a=a, parent=self, depth=self.depth+1, termiante=terminate)
        if termiante:
            estimated_return = r
        else:
            estimated_return = r+torch.max(self.children[action].Q)
        self.children[action].backpropagate(estimated_return)


    def backpropagate(self, estimated_return):
        if self.parent != None:
            old = self.parent[self.prev_a].tree_Q[self.prev_a]
            if estimated_return-1> old:
                self.parent[self.prev_a].tree_Q[self.prev_a] = estimated_return-1
                self.parent.backpropagate(new_estim_return)


if __name__=="__main__":
    n_moves = 1
    s = Cube().shuffle(n_moves)
    mcts = MCTS(root_state = s, max_depth=n_moves)