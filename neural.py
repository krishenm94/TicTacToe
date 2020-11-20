from player import Player
from board import Board, Cell, Result
from random_player import Random

import torch
from torch import nn
import numpy as np
from time import sleep
from tqdm import trange
from collections import deque

DISCOUNT_FACTOR = 1.0
INITIAL_EPSILON = 0.7
TRAINING_GAMES = 100000

class Net(nn.Module):
    """"docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.d11 = nn.Linear(9, 36)
        self.d12 = nn.Linear(36,36)
        self.output = nn.Linear(36, 9)

    def forward(self, x):
        x = self.d11(x)
        x = torch.relu(x)
        x = self.d12(x)
        x = torch.relu(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x


class Neural(Player):
    """docstring for Neural"""

    def __init__(self, optimizer, loss_function):
        super(Neural, self).__init__("Neural Network")
        self.policy_net = Net()
        self.target_net = Net()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optimizer
        self.loss_function = loss_function

    def get_best_move(self, board):
        net_output = self.get_target_net_output(board)
        valid_output_move_pairs = self.filter_output(net_output, board)
        best_move, _ = max(valid_output_move_pairs, key=lambda pair : pair[1])
        return best_move

    def get_target_net_output(self, board):
        net_input = torch.tensor(board.cells, dtype=torch.float)
        return self.target_net(net_input)

    @staticmethod
    def filter_output(self, net_output, board):
        valid_moves = board.get_valid_moves()
        valid_output_move_pairs = []
        for move in valid_moves:
            valid_output_move_pairs.append((move, net_output[move].item()))
        return valid_output_move_pairs

    def train(self, turn, opponent=Random(), total_games=TRAINING_GAMES):
        print(f"Training {self.name} for {total_games} games.", flush=True)
        self.turn = turn
        opponent.set_turn(self.turn % 2 + 1)
        epsilon = INITIAL_EPSILON

        sleep(0.05)  # Ensures no collisions between tqdm prints and main prints
        for game in trange(total_games):

            self.play_training_game(opponent, epsilon)
            # Decrease exploration probability
            if (game + 1) % (total_games / 10) == 0:
                epsilon = max(0, epsilon - 0.1)
                # tqdm.write(f"{game + 1}/{total_games} games, using epsilon={epsilon}...")

    def play_training_games(self, opponent, epsilon):
        move_history = deque()
        board = Board()
        x_player = self if self.turn == 1 else opponent
        o_player = self if self.turn == 2 else opponent

        while not board.is_game_over():
            player = o_player
            if board.whose_turn() == Cell.X:
                player = x_player

            if player is self:
                board = self.training_move(board, epsilon, move_history)
            else:
                player.move(board)

        self.post_training_game_update(board, move_history)

    def post_training_game_update(self, board, move_history):

