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

class ReplayMemory():
    def __init__(self, maxlen):
        self.deque = collections.deque(maxlen=maxlen)
        self.maxlen = maxlen
    

    def add(self, sars):
        if len(self.deque) >= self.maxlen:
            self.deque.popleft()
        self.deque.append(sars)

    
    def sample(self, batch_size=64):
        if len(self.deque) < batch_size:
            size = len(self.deque)
        else:
            size = batch_size
        samples = random.sample(self.deque, size)
        return samples


def cube_to_tensor(s):
    tensor = torch.from_numpy(s.cube_array)
    return tensor.to(device).view(1, -1)
    #return tensor.to(device).view(1, 6*6, 3, 3)


def eps_greedy(network, s, eps):
    tensor = cube_to_tensor(s)
    result = network(tensor)
    if np.random.rand() > eps:
        q, a = torch.max(result, dim=1)
    else:
        a = np.random.randint(len(s.actions))
        a = torch.tensor([a], dtype=torch.long, device=device)
        q = result[0,a]
    return q, a, tensor


def step(s, a):
    s_prime = s.copy().take_action(a)
    terminate = s_prime.check_if_solved()
    if terminate:
        r = 40
    else:
        r = -1
    return s_prime, r, terminate


def decrease_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("updated learning rate: new lr:", param_group['lr']/10)
        param_group['lr'] = param_group['lr']/10


def DQN():
    gamma = 0.95
    n_moves = 2
    QNet = FCC2x2().to(device)
    QNet.load_state_dict(torch.load("savedir/QNet_60k.pth"))
    QNet_fixed = FCC2x2().to(device)
    QNet_fixed.eval()
    optimizer = optim.Adam(QNet.parameters(), lr=0.001)
    #optimizer = optim.SGD(QNet.parameters(), lr=0.001, momentum = 0.9, weight_decay=0.0001)
    replay_memory = ReplayMemory(maxlen=10000) # [(r, Q_tar, Q)]
    returns = [] #[ep1, ep2, ep3, ...]

    num_episodes = 60001
    mod_update_target_network = 10000
    for episode in range(num_episodes):
        if episode % mod_update_target_network == 0:
            QNet_fixed.load_state_dict(QNet.state_dict())
            torch.save(QNet.state_dict(), "savedir/QNet_"+str(episode//1000)+"k.pth")
        if episode in []:
            decrease_lr(optimizer)

        s = Cube().shuffle(n_moves)
        ctr = 0
        terminate = False
        total_return = 0
        while not terminate:
            QNet.eval()
            eps = 0.05
            #eps= 0.01
            Q, a, s_tensor = eps_greedy(QNet, s, eps)
            s_prime, r, terminate = step(s, a)
            if not terminate:
                Q_tar, a, s_tensor_prime = eps_greedy(QNet_fixed, s_prime, eps=0)
            else:
                Q_tar = 0
                s_tensor_prime = None
            replay_memory.add([s_tensor, a, r , s_tensor_prime, terminate])

            QNet.train()
            optimize_model(replay_memory, QNet, QNet_fixed, optimizer, gamma, batch_size=128)

            s = s_prime
            total_return = r + gamma*total_return

            if episode % 100 == 0:
                print("episode", episode, "return", total_return)
                #print(Q)

            ctr +=1
            if ctr == n_moves:
                break

        returns += [total_return]
        
    graph(values2ewma(returns))

        

def optimize_model(replay_memory, QNet, QNet_fixed, optimizer, gamma, batch_size=128):
    samples = replay_memory.sample(batch_size)

    if len(samples) >1:

        s_tensor, a, r , s_tensor_prime, terminate = zip(*samples)

        Q = QNet(torch.cat(s_tensor, dim=0))
        Q = Q.gather(dim=1, index=torch.stack(a)).squeeze(1)


        Q_tar = torch.zeros(Q.size(0), device=device)
        not_none_s_prime = [s for s in s_tensor_prime if s is not None]
        if not not_none_s_prime == []:
            not_none_s_prime = torch.cat(not_none_s_prime)
            not_none_mask = torch.tensor([*map(lambda s: s is not None, s_tensor_prime)], device=device, dtype=torch.uint8)
            Q_tar[not_none_mask] = QNet_fixed(not_none_s_prime).max(dim=1)[0].detach()

        r = torch.tensor(r, dtype=torch.float, device=device)

        optimizer.zero_grad()
        loss = F.smooth_l1_loss(input=Q, target=(r+gamma*Q_tar), size_average=True, reduce=True)
        loss.backward()
        optimizer.step()

        global losses
        losses += [loss.detach().cpu().numpy()]


def main():
    pass

def test():
    cube = Cube()
    tensor = cube_to_tensor(cube)
    QNet = FCC().to(device)
    result = QNet(tensor)

    print(result)
    print(result.size())
    q, a = torch.max(result, dim=1)
    print("q", q)
    print("a", a)


if __name__ == "__main__":
    losses = []
    DQN()
    graph(values2ewma(losses, alpha=0.9))
    # replay_memory = ReplayMemory(maxlen=10)
    # for i in range(10000):
    #     replay_memory.add([i, i])


