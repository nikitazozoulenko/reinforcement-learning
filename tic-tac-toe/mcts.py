import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from environment import GameBoard, step


device = torch.device("cuda")

def eps_greedy(action_values, eps):
    '''Takes in a tensor of size [n_actions] and float eps. 
       Returns sorted list of actions chosen eps-greedily'''
    if np.random.rand() > eps:
        _q, a = torch.sort(action_values, descending=True)
    else:
        a = torch.randperm(action_values.size(-1))
    return a


class MCTS:
    def __init__(self, board_size, win_length, network): 
        self.network = network
        self.win_length = win_length
        self.board_size = board_size
        self.reset()


    def monte_carlo_tree_search(self, mcts_steps=100, eps=0.05):
        '''Does mcts_steps MCTS simulations eps-greedily.
        Returns best (most taken) action'''
        t=0
        while t<mcts_steps:
            self.root.single_simulation(eps)
            t += 1
        a = self.get_best_action()
        return a


    def get_best_action(self):
        '''Returns most taken action if deterministic play, else selects action proportional to n(s, a)'''
        if self.deterministic_play:
            a = torch.argmax(self.root.children_n_visited)
        else:
            probs = self.root.children_n_visited
            probs = (probs/torch.sum(probs)).cpu().numpy()
            a = np.random.choice(probs.shape[-1], size=1, p=probs)[0]
        return a

    
    def set_deterministic_play(self, deterministic_play):
        '''Takes in (bool) deterministic_play'''
        self.deterministic_play = deterministic_play

    
    def change_root_with_action(self, a):
        '''Replaces root with the node which you would get by taking action a'''
        if self.root.children[a] == None:
            next_root = self.root.expansion(a)
        else:
            next_root = self.root.children[a]
        next_root.parent = None
        self.root.next_root = next_root
        self.root = next_root
    
    
    def reset(self):
        '''Resets the MCTS tree to only root of starting game board'''
        self.root = Node(network=self.network, s=GameBoard(self.board_size, self.win_length), parent=None, prev_a=None, prev_r=None, terminate=False)
        self.original_root = self.root
        self.set_deterministic_play(False)
    

    def traverse_and_add_to_replay(self, experience_replay, gt_v, is_agents_turn=None, current=None):
        '''Recursively traverses the tree from original root and adds to experience replay'''
        #agents turns only
        if current == None:
            current = self.original_root
            is_agents_turn = torch.sum(self.original_root.children_n_visited) != 0
        
        if is_agents_turn:
            #NOTE: nope-assume that we always use non-deterministical target policies
            #NOTE: using temp 0.1 ish
            gt_probs = current.children_n_visited**3
            gt_probs = gt_probs/torch.sum(gt_probs)
            experience_replay.add([gt_probs, gt_v, current.s.to_tensor()[0]])

        if current.next_root != None:
            if current.next_root.next_root != None:
                is_agents_turn = not is_agents_turn
                self.traverse_and_add_to_replay(experience_replay, -gt_v, is_agents_turn, current.next_root)
        
        
def get_symmetry_results(network, s):
    board_array = s.board_array
    #TODO TODO
    probs, v = None, None

class Node:
    def __init__(self, network, s, parent, prev_a, prev_r, terminate):
        self.next_root = None
        self.network = network
        self.s = s
        self.parent = parent
        self.prev_a = prev_a
        self.prev_r = prev_r
        self.terminate = terminate
        self.children = [None] * self.s.size**2
        self.children_n_visited = torch.zeros(self.s.size**2).to(device)

        self.probs, self.v = self.network(self.s.to_tensor())
        self.probs = self.probs[0].data
        self.v = self.v[0].data
        self.Q = torch.zeros(self.probs.size()).to(device)


    def single_simulation(self, eps):
        '''Add a single MCTS simulation to the tree'''
        expanded_node = self.selection_and_expansion(eps)
        v = expanded_node.evaluation()
        expanded_node.backup(v)


    def selection_and_expansion(self, eps):
        '''Traverses the tree based on UCT (eps greedily)
           Returns expanded node'''
        if self.terminate:
            return self

        total_n = torch.sum(self.children_n_visited)
        if total_n == 0:
            total_n+=1

        #recursively search for best leaf node
        c = np.sqrt(2)
        U = self.probs*torch.sqrt(total_n)/(self.children_n_visited+1)*c
        sorted_actions = eps_greedy(self.Q + U, eps)
        for a in sorted_actions:
            if self.s.check_if_legal_action(a):
                if self.children[a] != None:
                    return self.children[a].selection_and_expansion(eps)
                else:
                    return self.expansion(a)
        
        #the program should never get to this line
        print(self.s)
        raise RuntimeError("NO ACTIONS WERE LEGAL")


    def expansion(self, a):
        '''Expands tree from current leaf node with action a.
           Returns expanded node'''
        s_prime, r, terminate = step(self.s, a)
        self.children[a] = Node(self.network, s=s_prime.reverse_player_positions(), parent=self, prev_a=a, prev_r=r, terminate=terminate)
        return self.children[a]


    def evaluation(self):
        '''Evaluates current state (some is already done in __init__)
           Returns evaluated state value'''
        if self.terminate:
            self.v = -self.prev_r
        return self.v


    def backup(self, v):
        '''Backs up tree statistics up to root
           Currently uses mean Q'''
        if self.parent != None:
            n_a = self.parent.children_n_visited[self.prev_a]
            self.parent.children_n_visited[self.prev_a] = n_a + 1
            n = torch.sum(self.parent.children_n_visited) - 1

            old_Q = self.parent.Q[self.prev_a]
            new_Q = (-v+old_Q*n)/(n+1)
            self.parent.Q[self.prev_a] = new_Q
            self.parent.backup(-v)


if __name__=="__main__":
    from network import CNN
    network = CNN().to(device)
    network.eval()
    mcts = MCTS(3, 3, network)
    a = mcts.monte_carlo_tree_search(100, 0.05)
    print(a)
