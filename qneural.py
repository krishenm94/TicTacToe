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


class QNeural(Player):
    """docstring for QNeural"""

    def __init__(self, optimizer, loss_function):
        super(QNeural, self).__init__("Q Neural Network")
        self.policy_net = Net()
        self.target_net = Net()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optimizer
        self.loss_function = loss_function

    def get_best_move(self, board):
        net_output = self.get_q_values(board, self.target_net)
        valid_output_move_pairs = self.filter_output(net_output, board)
        best_move, _ = max(valid_output_move_pairs, key=lambda pair : pair[1])
        return best_move

    @staticmethod
    def get_q_values(self, board, net):
        net_input = torch.tensor(board.cells, dtype=torch.float)
        return net(net_input)

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

    def training_move(self, board, epsilon, move_history):
        move = self.choose_move_index(board, epsilon)
        move_history.appendleft((board, move))
        return board.simulate_turn(move)

    def post_training_game_update(self, board, move_history):
        end_state_value = self.get_end_state_value(board)

        # Initial loss update
        next_board, move = move_history[0]
        self.backpropagate(next_board, move, end_state_value)

        for board, move in move_history[1:]:
            with torch.no_grad():
                next_q_values = self.get_q_values(next_board, self.target_net)
                max_next_q_value = torch.max(next_q_values).item()

            self.backpropagate(board, move, max_next_q_value * DISCOUNT_FACTOR)
            next_board = board

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def backpropagate(self, board, move, target_value):
        self.optimizer.zero_grad()

        board_tensor = torch.tensor(board.cells, dtype=torch.float)
        output = self.policy_net(board_tensor)

        target_output = output.clone().detach()

        for move in board.get_invalid_moves():
            target_output[move] = -1

        loss = self.loss_function(output, target_output)
        loss.backward()

        self.optimizer.step()

    def get_end_state_value(self, board):
        assert board.is_game_over(), "Game is not over"

        game_result = board.get_game_result()

        if game_result == Result.Draw:
            return 0

        if game_result == Result.X_Wins:
            return 1 if self.turn == 1 else -1

        if game_result == Result.O_Wins:
            return 1 if self.turn == 2 else -1

        assert False, "Undefined behaviour"