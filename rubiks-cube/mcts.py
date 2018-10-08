import numpy as np
import torch
import torch.nn.functional as F

from cube2x2 import Cube, step, cube_to_tensor


device = torch.device("cuda")

def eps_greedy(action_values, eps):
    '''Returns sorted list of actions chosen eps-greedily'''
    if np.random.rand() > eps:
        _q, a = torch.sort(action_values, descending=True)
    else:
        a = torch.randperm(action_values.size(-1))
    return a[0]


class MCTS:
    def __init__(self, network): 
        self.network = network
        self.reset(n_shuffle=0)


    def monte_carlo_tree_search(self, max_steps=100, mcts_eps=0.1, final_choose_eps=0.1):
        t=0
        while t<max_steps:
            self.root.uct_traverse(mcts_eps)
            t += 1
        a = self.get_best_action(final_choose_eps)
        self.change_root_with_action(a)
        return a, self.root.is_terminate_state

    
    def change_root_with_action(self, a):
        if self.root.children[a] == None:
            self.root.simulate(a)
        self.root = self.root.children[a]


    def get_best_action(self, eps):
        a = self.root.best_action(eps)
        return a


    def reset(self, n_shuffle):
        s = Cube()
        s.shuffle(n_shuffle)
        self.root = Node(network=self.network, s=s, parent=None, prev_a=None, prev_r=None, is_terminate_state=s.check_if_solved())

    
    def get_original_root(self):
        node = self.root
        while node.parent != None:
            node = node.parent
        return node
        

class Node:
    def __init__(self, network, s, parent, prev_a, prev_r, is_terminate_state):
        self.network = network
        self.s = s
        self.parent = parent
        self.prev_a = prev_a
        self.prev_r = prev_r
        self.is_terminate_state = is_terminate_state

        self.tree_Q = torch.mean(self.network(cube_to_tensor(s)), dim=0)
        self.children = [None] * self.tree_Q.size(-1)

        self.n_visited = 1
    

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
        # U = np.sqrt(2*np.log(self.n_visited)/children_n_visited).astype(np.float32)
        # a = eps_greedy(self.tree_Q + torch.from_numpy(U).to(device), eps)
        a = eps_greedy(self.tree_Q, eps)
        return a
        
        #the program should never get to this line
        raise RuntimeError("NO ACTIONS WERE LEGAL")

    
    def simulate(self, a):
        s_prime, r, terminate = step(self.s, a)
        self.children[a] = Node(s=s_prime, network=self.network, parent=self, prev_a=a, prev_r=r, is_terminate_state=terminate)
        if terminate:
            estimated_return = torch.tensor(r).float().to(device)
            self.children[a].tree_Q.zero_()
        else:
            estimated_return = r+torch.max(self.children[a].tree_Q.data)
        self.tree_Q[a] = estimated_return
        self.backpropagate()
        

    def backpropagate(self):
        if self.parent != None:
            old_q_value = self.parent.tree_Q[self.prev_a]
            new_q_value = self.prev_r+torch.max(self.tree_Q)
            if old_q_value != new_q_value:
                self.parent.tree_Q[self.prev_a] = new_q_value
                self.parent.backpropagate()


if __name__=="__main__":
    pass
