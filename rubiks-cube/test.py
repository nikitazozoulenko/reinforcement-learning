import numpy as np
import torch

def eps_greedy(action_values, eps):
    if np.random.rand() > eps:
        q, a = torch.max(action_values, dim=-1)
    else:
        a = np.random.randint(action_values.size(-1))
        a = torch.tensor([a], dtype=torch.long, device=device)
        q = result[0,a]
    return q, a

if __name__ == "__main__":
    device = torch.device("cuda")
    Q = torch.ones((1,12))
    children = [True] * Q.size(-1)
    selfvisited = 2.7272
    Q[0,3] += 1

    q, a = torch.sort(Q, dim=-1, descending=True)
    print(q)
    print(a[0].size())
    maxx = torch.max(Q)
    print(maxx, maxx.size())