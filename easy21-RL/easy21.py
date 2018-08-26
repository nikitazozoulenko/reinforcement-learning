import numpy as np
import matplotlib.pyplot as plt


def values2ewma(losses, alpha = 0.999):
    losses_ewma = []
    ewma = losses[0]
    ewma = 0
    for loss in losses:
        ewma = alpha*ewma + (1-alpha)*loss
        losses_ewma += [ewma]
    return losses_ewma

def graph(rewards):
    plt.plot(rewards, "b", label = "Reward")
    plt.legend(loc=1)
    plt.show()


class State():
    def __init__(self, player_sum=0, dealer_sum=0):
        self.player_sum = player_sum
        self.dealer_sum = dealer_sum
    
    def to_index(self):
        return self.player_sum + 21*(self.dealer_sum-1) -1
    

def draw_card(only_black=False):
    card = np.random.randint(1,11)
    if not only_black and np.random.rand() > 2/3:
        card *= -1
    return card


def init_state():
    state = State(draw_card(only_black=True), draw_card(only_black=True))
    return state


def step(state, a): #a=0 stick, a=1 hit
    if a == 1: #stick
        state.player_sum += draw_card()
        if state.player_sum > 21 or state.player_sum < 1:
            terminate = True
            r = -1
        else:
            terminate = False
            r = 0
    elif a == 0:
        terminate = False
        while not terminate:
            if state.dealer_sum > 21 or state.dealer_sum < 1:
                terminate = True
                r = 1
            elif state.dealer_sum < 17:
                state.dealer_sum += draw_card()
            else:
                terminate = True
                if state.player_sum > state.dealer_sum:
                    r = 1
                elif state.player_sum < state.dealer_sum:
                    r = -1
                else:
                    r = 0

    return r, state, terminate


def monte_carlo_control():
    Q = np.zeros((10*21, 2)) # Q(s, a)
    all_rewards = []
    N_0 = 100
    N_s = np.zeros((10*21))
    N_s_a = np.zeros((10*21, 2))

    num_episodes = 500000
    for episode in range(num_episodes):
        #init
        state = init_state()
        state_action_pairs = []
        rewards = []
        G = [] #Gt0 = rt1 + g*rt2 + g*g*rt3 + ...

        #play episode
        terminate = False
        while not terminate:
            #eps-greedily select action
            s = state.to_index()
            eps = N_0 / (N_0 + N_s[s])
            if np.random.rand() > eps:
                a = np.argmax(Q[s])
            else:
                a = np.random.randint(len(Q[s]))
            
            #step
            N_s[s] += 1
            N_s_a[s, a] += 1
            state_action_pairs.append([s, a])
            r, state, terminate = step(state, a)
            rewards.append(r)
        
        #learn from all state action pairs
        gamma = 1
        for t, r in enumerate(rewards):
            Gt = 0
            for i in range(len(rewards)-t):
                Gt += gamma**i * rewards[i]     
            G.append(Gt)           

        for t, s_a_pair in enumerate(state_action_pairs):
            s, a = s_a_pair
            Q[s, a] = Q[s, a] + 1/N_s_a[s, a]*(G[t]-Q[s,a])
        all_rewards.append(G[-1])
    print(Q)
    graph(values2ewma(all_rewards))


def eps_greedy(Q, s, eps):
    if np.random.rand() > eps:
        a = np.argmax(Q[s])
    else:
        a = np.random.randint(len(Q[s]))
    return a


def sarsa_lambda(lmbda = 0.9, gamma = 1):
    Q = np.zeros((10*21, 2)) # Q(s, a)
    all_rewards = []
    N_0 = 100
    N_s = np.ones((10*21))
    N_s_a = np.ones((10*21, 2))

    num_episodes = 500000
    for episode in range(num_episodes):
        #init
        total_reward = 0
        E = np.zeros((10*21, 2))
        state = init_state()
        s = state.to_index()
        eps = N_0 / (N_0 + N_s[s])
        a = eps_greedy(Q, s, eps)

        #play episode
        terminate = False
        while not terminate:
            r, state, terminate = step(state, a)
            s_prime = state.to_index()
            eps = N_0 / (N_0 + N_s[s])
            if not terminate:
                a_prime = eps_greedy(Q, s_prime, eps)
                delta = r + gamma*Q[s_prime, a_prime] - Q[s, a]
            else:
                delta = r - Q[s,a]
            E[s, a] += 1
            N_s[s] +=1
            N_s_a[s, a] +=1

            #for all states
            Q = Q + np.reciprocal(N_s_a) * delta*E
            E = gamma*lmbda*E

            if not terminate:
                s = s_prime
                a = a_prime

            #track stats
            total_reward = r + gamma*total_reward
        all_rewards.append(total_reward)
    graph(values2ewma(all_rewards, alpha = 0.999))


if __name__ == "__main__":
    monte_carlo_control()
    #sarsa_lambda(lmbda=0.9, gamma=1)