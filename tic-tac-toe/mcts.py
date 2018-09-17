import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from network import FCC
from environment import GameBoard


device = torch.device("cuda")
QNet = FCC().to(device)
QNet.eval()


def board_to_tensor(s):
    tensor = torch.from_numpy(s.board)
    return tensor.to(device).view(1, -1)


def eps_greedy(action_values, eps):
    '''Returns sorted list of actions chosen eps-greedily'''
    if eps == 0 or np.random.rand() > eps:
        _q, a = torch.sort(action_values, descending=True)
    else:
        a = torch.randperm(action_values.size())
    return a


def step(s, a):
    s_prime = s.copy().take_action(a)
    terminate = s_prime.check_win_position()
    if terminate:
        r = 1
    else:
        r = 0
    return s_prime, r, terminate


class MCTS:
    def __init__(self, root_state):
        self.root = Node(s=root_state, parent=None, depth=0, is_terminate_state=False)
        self.t = 0


    def monte_carlo_tree_search(self, max_steps=1000, choose_eps=0):
        while self.t<1:
            self.root.uct_traverse()
            self.t +=1

        a = self.best_action(choose_eps)
        return a


    def best_action(self, eps):
        return self.root.best_action(choose_eps)
        


class Node:
    def __init__(self, s, parent, depth, is_terminate_state):
        self.s = s
        self.parent = parent
        self.depth = depth
        self.is_terminate_state = is_terminate_state

        self.Q = QNet(board_to_tensor(s))
        self.tree_Q = self.Q[0]
        self.children = [None] * self.Q.size(-1)

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

        #the program should never get to this line
        raise RuntimeError("NO ACTIONS WERE LEGAL")


    def best_action(self, eps):
        '''Returns the best action to pick. Also checks that the move is legal'''
        #Get statistics for UCT formula
        children_n_visited = np.ones((self.Q.size(-1)))
        for i, child in enumerate(self.children):
            if child != None:
                children_n_visited[i] = child.n_visited
        U = np.sqrt(2*np.log(self.n_visited)/children_n_visited).astype(np.float32)
        _, sorted_actions = eps_greedy(self.tree_Q + torch.from_numpy(U).to(device), eps)
        for a in sorted_actions:
            if self.s.check_if_legal_move(a):
                return a

    
    def simulate(self, a):
        s_prime, r, terminate = step(self.s, a)
        self.children[a] = Node(s=s_prime.reverse_player_positions(), parent=self, depth=self.depth+1)
        if termiante:
            estimated_return = r
        else:
            estimated_return = r+torch.max(self.children[a].tree_Q)
        assert True == False
        self.children[action].backpropagate(estimated_return)
        


if __name__=="__main__":
    s = GameBoard(size=3, win_length=3)
    mcts = MCTS(root_state = s)
    mcts.monte_carlo_tree_search()












# class OLDOLDOLDOLDOLDNode:
#     def __init__(self, s, prev_a, parent, depth, terminate):
#         self.parent = parent
#         self.s = s
#         self.prev_a=prev_a
#         self.depth=depth
#         self.terminate=terminate
#         self.Q = QNet(cube_to_tensor(s))
#         self.tree_Q = np.zeros(self.Q.size(-1)) #torch.zeros((1, self.Q.size(-1)), device=device)
#         self.children = [None] * self.Q.size(-1)
#         self.visited = 1


#     def uct_traverse(self):
#         children_n_visit = []
#         for child in self.children:
#             if child == None:
#                 return self
#             children_n_visit += [child.visited]
#         children_n_visit = torch.tensor(visited, device=device)

#         U = (np.log(self.visited) / children_n_visit) ** 0.5
#         q, a = eps_greedy(self.Q+U, eps=0.01)
#         return self.children[a].uct_traverse()


#     def simulate(self, eps):
#         #########################TODO TODO TODO eps not implemented yet TODO TODO TODO############################
#         q, a = torch.sort(Q, dim=-1, descending=True)
#         for action in a[0]:
#             if self.children[action] != None:
#                 a = action
#                 break
        
#         s_prime, r, terminate = step(self.s, a)
#         self.children[action] = Node(s=s_prime, prev_a=a, parent=self, depth=self.depth+1, termiante=terminate)
#         if termiante:
#             estimated_return = r
#         else:
#             estimated_return = r+torch.max(self.children[action].Q)
#         self.children[action].backpropagate(estimated_return)


#     def backpropagate(self, estimated_return):
#         if self.parent != None:
#             old = self.parent[self.prev_a].tree_Q[self.prev_a]
#             if estimated_return-1> old:
#                 self.parent[self.prev_a].tree_Q[self.prev_a] = estimated_return-1
#                 self.parent.backpropagate(new_estim_return)

