import random
import collections
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network import FCC
from environment import GameBoard, step
from mcts import MCTS, eps_greedy, board_to_tensor
from graph import graph, ewma

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

    
    def sample(self, batch_size=64):
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

    
    # def add_to_experience_replay(self):
    #     s = board_to_tensor(self.mcts.root.s)[0]
    #     tree_Q = self.mcts.root.tree_Q.data
    #     allowed_actions = self.mcts.root.allowed_actions
    #     self.experience_replay.add([s, tree_Q, allowed_actions])


    def add_to_experience_replay(self, node):
        s = board_to_tensor(node.s)[0]
        tree_Q = node.tree_Q.data
        allowed_actions = node.allowed_actions
        self.experience_replay.add([s, tree_Q, allowed_actions])


class MatchHandler:
    def __init__(self, agent, opponent, game_board):
        self.agent = agent
        self.opponent = opponent
        self.game_board = game_board
        self.reset_tally_results()

    
    def play_match(self, max_mcts_steps, mcts_eps, final_choose_eps, do_print=False):
        #randomize player start
        self.game_board.reset()
        self.agent.mcts.reset()
        self.opponent.mcts.reset()
        players = [self.agent, self.opponent]
        agent_starts = bool(random.randint(0,1))
        if not agent_starts:
            players.reverse()
        if do_print:
            if agent_starts:
                print("AGENT STARTS")
            else:
                print("OPPONENT STARTS")

        #turn loop until terminate
        _ = players[1].monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps) #player waiting gets to start thinking first
        while True:
            end_match, coords = self.play_turn(players[0], players[1], max_mcts_steps, mcts_eps, final_choose_eps, do_print)
            if end_match:
                break
            players.reverse()

        #evaluate who won and tally results
        if coords:
            cross_won = self.game_board.board[coords[0]] > 0
            if cross_won == agent_starts:
                self.wins_agent += 1
            else:
                self.losses_agent += 1
        else:
            self.draws_agent += 1

        #traverse tree backwards and add to experience replay
        self.traverse_and_add_to_replay()


    def traverse_and_add_to_replay(self):
        node = self.agent.mcts.root
        while node.parent != None:
            node = node.parent
            self.agent.add_to_experience_replay(node)
        self.agent.add_to_experience_replay(node)


    def play_turn(self, player, opponent, max_mcts_steps, mcts_eps, final_choose_eps, do_print):
        picked_action = player.monte_carlo_tree_search(max_mcts_steps, mcts_eps, final_choose_eps)
        self.game_board.take_action(picked_action)
        player.change_root_with_action(picked_action)
        opponent.change_root_with_action(picked_action)
        # for person in [player, opponent]:
        #     if person.experience_replay != None:
        #         person.add_to_experience_replay()
        #     person.change_root_with_action(picked_action)
        terminate, coords = self.game_board.check_win_position()

        if do_print:                
            print("game board")
            print(terminate, coords)
            print(self.game_board)
            print(player.mcts.root.tree_Q)
            print()
        return terminate, coords


    def reset_tally_results(self):
        self.wins_agent = 0
        self.draws_agent = 0
        self.losses_agent = 0
    

class OptimizerHandler:
    def __init__(self, match_handler, batch_size, n_iter_train, learning_rate):
        self.match_handler=match_handler
        self.batch_size = batch_size
        self.n_iter_train = n_iter_train
        self.learning_rate = learning_rate
        self.reset_optim_after_opponent_update = True
        self.MSE = nn.MSELoss()
        self.create_optim()

        self.optim_counter = 0
        self.losses = []
        

    def create_optim(self):
        self.optimizer = optim.SGD(self.match_handler.agent.mcts.network.parameters(), weight_decay=0.001, lr=self.learning_rate, momentum=0.9, nesterov=True)
        # self.optimizer = optim.Adam(self.match_handler.agent.mcts.network.parameters(), weight_decay=0.001, lr=self.learning_rate)


    def optimize_model(self):
        model = self.match_handler.agent.mcts.network
        experience_replay = self.match_handler.agent.experience_replay
        model.train()
        first_loss = None
        for _ in range(self.n_iter_train):
            samples = experience_replay.sample(self.batch_size)
            s, target, allowed_a = zip(*samples)
            s = torch.stack(s)
            target = torch.stack(target)
            allowed_a = torch.from_numpy(np.stack(allowed_a))#.to(device)

            self.optimizer.zero_grad()
            loss = self.MSE(model(s)[allowed_a], target[allowed_a])
            first_loss = loss.data.cpu().numpy() if first_loss == None else first_loss
            loss.backward()
            self.optimizer.step()
        model.eval()

        self.losses += [(self.optim_counter, first_loss, loss.data.cpu().numpy())]
        self.optim_counter += 1

    
    def update_opponent_if_needed(self, min_n_games=500, max_n_games=2000):
        wins = self.match_handler.wins_agent
        draws = self.match_handler.draws_agent
        losses = self.match_handler.losses_agent
        n_games = wins+draws+losses
        if n_games > min_n_games:
            if wins/(wins+losses) > 0.60:
                self.update_opponent()
            elif n_games > max_n_games:
                if wins/(wins+losses) > 0.5:
                    self.update_opponent()
                elif wins/(wins+losses) < 0.5:
                    self.reset_agent_to_last_save()


    def reset_agent_to_last_save(self):
        agent_network = self.match_handler.agent.mcts.network
        opponent_network = self.match_handler.opponent.mcts.network
        opponent_network.load_state_dict(torch.load(model_path))
        agent_network.load_state_dict(torch.load(model_path))
        self.match_handler.reset_tally_results()
        if self.reset_optim_after_opponent_update:
            self.create_optim()
            self.match_handler.agent.experience_replay.reset()
    

    def update_opponent(self):
        agent_network = self.match_handler.agent.mcts.network
        opponent_network = self.match_handler.opponent.mcts.network
        torch.save(agent_network.state_dict(), model_path)
        opponent_network.load_state_dict(torch.load(model_path))
        self.match_handler.reset_tally_results()
        if self.reset_optim_after_opponent_update:
            self.create_optim()
            self.match_handler.agent.experience_replay.reset()


def create_agent_and_opponent(board_size, win_length, replay_maxlen):
    if not os.path.exists(model_path):
        torch.save(FCC(board_size).to(device).state_dict(), model_path)

    #opponent
    opponent_network = FCC(board_size).to(device)
    opponent_network.load_state_dict(torch.load(model_path))
    opponent_network.eval()
    opponent_mcts = MCTS(board_size, win_length, opponent_network)
    opponent = Player(opponent_mcts, None)

    #agent
    agent_network = FCC(board_size).to(device)
    agent_network.load_state_dict(torch.load(model_path))
    agent_network.eval()
    agent_mcts = MCTS(board_size, win_length, agent_network)
    agent = Player(agent_mcts, ExperienceReplay(replay_maxlen))
    return agent, opponent, GameBoard(board_size, win_length)


def main():
    #variables
    board_size = 3
    win_length = 3
    max_mcts_steps=10
    mcts_eps=0.05
    final_choose_eps=0
    replay_maxlen = 2500
    batch_size = 256
    n_iter_train = 10
    learning_rate = 0.001
    min_n_games=100
    max_n_games=200

    #match handler
    match_handler = MatchHandler(*create_agent_and_opponent(board_size, win_length, replay_maxlen))
    optimizer_handler = OptimizerHandler(match_handler, batch_size, n_iter_train, learning_rate)

    #play some games
    for i in range(100000):
        match_handler.play_match(max_mcts_steps, mcts_eps, final_choose_eps)
        optimizer_handler.optimize_model()
        print()
        print("wins", match_handler.wins_agent)
        print("draws", match_handler.draws_agent)
        print("losses", match_handler.losses_agent)
        print("LOSS", optimizer_handler.losses[-1][-1])
        optimizer_handler.update_opponent_if_needed(min_n_games, max_n_games)

        if not i % 100 and i:
            graph(ewma(np.array(optimizer_handler.losses)[:, -1], alpha=0))



if __name__ == "__main__":
    main()


