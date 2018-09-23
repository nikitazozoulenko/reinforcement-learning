import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from network import FCC
from environment import GameBoard, step


device = torch.device("cuda")

def board_to_tensor(s):
    tensor = torch.from_numpy(s.board)
    return tensor.to(device).view(1, -1)


def eps_greedy(action_values, eps):
    '''Returns sorted list of actions chosen eps-greedily'''
    if eps == 0 or np.random.rand() > eps:
        _q, a = torch.sort(action_values, descending=True)
    else:
        a = torch.randperm(action_values.size(-1))
    return a


class MCTS:
    def __init__(self, board_size, win_length, network): 
        self.network = network
        self.win_length = win_length
        self.board_size = board_size
        self.root = Node(network=self.network, s=GameBoard(board_size, win_length), parent=None, prev_a=None, depth=0, is_terminate_state=False)


    def monte_carlo_tree_search(self, max_steps=100, mcts_eps=0.1, final_choose_eps=0.1):
        t=0
        while t<max_steps:
            self.root.uct_traverse(mcts_eps)
            t += 1
        a = self.get_best_action(final_choose_eps)
        return a

    
    def get_experience_replay_item(self):
        return self.root.tree_Q, self.root.s

    
    def change_root_with_action(self, a):
        if self.root.children[a] == None:
            self.root.simulate(a)
        self.root = self.root.children[a]


    def get_best_action(self, eps):
        a = self.root.best_action(eps)
        return a


    def reset(self):
        self.root = Node(network=self.network, s=GameBoard(self.board_size, self.win_length), parent=None, prev_a=None, depth=0, is_terminate_state=False)
        

class Node:
    def __init__(self, network, s, parent, prev_a, depth, is_terminate_state):
        self.network = network
        self.s = s
        self.parent = parent
        self.prev_a = prev_a
        self.depth = depth
        self.is_terminate_state = is_terminate_state

        self.tree_Q = self.network(board_to_tensor(s))[0]
        self.children = [None] * self.tree_Q.size(-1)

        self.n_visited = 1

        if depth == 0:
            self.player_modulo = 1 # TODO TODO TODO
    

    def uct_traverse(self, eps=0.1):
        '''Recursively returns new simulated node picked with eps-greedy UCT'''
        #check if terminate and enumerate
        self.n_visited += 1
        if self.is_terminate_state:
            return self

        #Get best legal action
        a = self.best_action(eps)

        #recursively returns the new simulated node.
        if self.children[a] == None:
            return self.simulate(a)
        else:
            return self.children[a].uct_traverse()


    def best_action(self, eps):
        '''Returns the best action to pick. Also checks that the move is legal'''
        #Get statistics for UCT formula
        children_n_visited = np.ones((self.tree_Q.size(-1)))
        for i, child in enumerate(self.children):
            if child != None:
                children_n_visited[i] = child.n_visited
        #U = np.sqrt(2*np.log(self.n_visited)/children_n_visited).astype(np.float32)
        #sorted_actions = eps_greedy(self.tree_Q + torch.from_numpy(U).to(device), eps)
        sorted_actions = eps_greedy(self.tree_Q, eps)
        for a in sorted_actions:
            if self.s.check_if_legal_action(a):
                return a
        
        #the program should never get to this line
        raise RuntimeError("NO ACTIONS WERE LEGAL")

    
    def simulate(self, a):
        s_prime, r, terminate = step(self.s, a)
        self.children[a] = Node(s=s_prime.reverse_player_positions(), network=self.network, parent=self, prev_a=a, depth=self.depth+1, is_terminate_state=terminate)
        if terminate:
            estimated_return = torch.tensor(r).float().to(device)
            #self.children[a].tree_Q.zero_()
        else:
            estimated_return = r-torch.max(self.children[a].tree_Q.data)
        self.tree_Q[a] = estimated_return
        self.backpropagate()
        

    def backpropagate(self):
        if self.parent != None:
            old_q_value = self.parent.tree_Q[self.prev_a]
            new_q_value = -torch.max(self.tree_Q)
            if old_q_value != new_q_value:
                self.parent.tree_Q[self.prev_a] = new_q_value
                self.parent.backpropagate()


    def iterate_print(self):
        print(self.depth)
        for child in self.children:
            if child != None:
                child.iterate_print()


if __name__=="__main__":
    pass
