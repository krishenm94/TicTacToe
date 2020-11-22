from player import Player
from board import Board, Cell, Result
from random_player import Random

import torch
from torch import nn
import torch.cuda as cuda
import numpy as np
from time import sleep
from tqdm import trange
from collections import deque
import random
from enum import Enum

DISCOUNT_FACTOR = 1.0
INITIAL_EPSILON = 0.7
TRAINING_GAMES = 1000000

PATH = "./checkpoint"

class Key(Enum):
    Net = 0,
    Optimizer = 1,
    Games = 2,
    Wins = 3,
    Draws = 4,
    Losses = 5

class QNeural(Player):
    """docstring for QNeural"""

    class Net(nn.Module):
        """"docstring for Net"""

        def __init__(self):
            super(QNeural.Net, self).__init__()
            self.d11 = nn.Linear(9, 36)
            self.d12 = nn.Linear(36, 36)
            self.output = nn.Linear(36, 9)

        def forward(self, x):
            x = self.d11(x)
            x = torch.relu(x)
            x = self.d12(x)
            x = torch.relu(x)
            x = self.output(x)
            x = torch.sigmoid(x)
            return x

    def __init__(self, loss_function):
        super(QNeural, self).__init__("Q Neural Network")

        # print(f"Nets loaded on {cuda.get_device_name(0)}")
        # print(cuda.get_device_properties(0))
        self.online_net = QNeural.Net()
        self.online_net.train()

        self.target_net = QNeural.Net()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.SGD(self.online_net.parameters(), lr=0.1)
        self.loss_function = loss_function

    def get_best_move(self, board):
        return self.choose_move_index(board, 0)

    def get_q_values(self, board, net):
        net_input = torch.tensor(board.cells, dtype=torch.float)
        return net(net_input)

    def filter_output(self, net_output, board):
        valid_moves = board.get_valid_moves()
        valid_output_move_pairs = []
        for move in valid_moves:
            valid_output_move_pairs.append((move, net_output[move].item()))
        return valid_output_move_pairs

    def train(self, turn, opponent=Random(), total_games=TRAINING_GAMES):
        print(f"Training {self.name} for {total_games} games.", flush=True)
        print(f"Starting game number: {self.games}")
        self.turn = turn
        opponent.set_turn(self.turn % 2 + 1)
        epsilon = INITIAL_EPSILON

        sleep(0.05)  # Ensures no collisions between tqdm prints and main prints
        for game in trange(total_games):
            self.games += 1
            self.play_training_game(opponent, epsilon)
            # Decrease exploration probability
            if (game + 1) % (total_games / 20) == 0:
                epsilon = max(0, epsilon - 0.05)
                # tqdm.write(f"{game + 1}/{total_games} games, using epsilon={epsilon}...")

            if (game + 1) % 10000 == 0:
                self.save()

    def save(self):
        torch.save({Key.Games.name: self.games,
                    Key.Wins.name: self.wins,
                    Key.Draws.name: self.draws,
                    Key.Losses.name: self.losses,
                    Key.Net.name: self.online_net.state_dict(),
                    Key.Optimizer.name: self.optimizer.state_dict()},
                   PATH + '_' + str(self.games))

    def play_training_game(self, opponent, epsilon):
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

    def choose_move_index(self, board, epsilon):
        if epsilon > 0:
            random_value_from_0_to_1 = np.random.uniform()
            if random_value_from_0_to_1 < epsilon:
                return random.choice(board.get_valid_moves())

        net_output = self.get_q_values(board, self.online_net)
        valid_move_value_pairs = self.filter_output(net_output, board)

        return max(valid_move_value_pairs, key=lambda pair: pair[1])[0]

    def post_training_game_update(self, board, move_history):
        end_state_value = self.get_end_state_value(board)

        # Initial loss update
        next_board, move = move_history[0]
        self.backpropagate(next_board, move, end_state_value)

        for board, move in list(move_history)[1:]:
            with torch.no_grad():
                # next_q_values = self.get_q_values(next_board, self.online_net) # QN
                next_q_values = self.get_q_values(next_board, self.target_net)  # Double QN
                max_next_q_value = torch.max(next_q_values).item()

            self.backpropagate(board, move, max_next_q_value * DISCOUNT_FACTOR)
            next_board = board

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

    def backpropagate(self, board, move, target_value):
        self.optimizer.zero_grad()

        board_tensor = torch.tensor(board.cells, dtype=torch.float)
        output = self.online_net(board_tensor)

        target_output = output.clone().detach()
        target_output[move] = target_value
        for move in board.get_invalid_moves():
            target_output[move] = -1

        loss = self.loss_function(output, target_output)
        loss.backward()

        self.optimizer.step()

    def get_end_state_value(self, board):
        assert board.is_game_over(), "Game is not over"

        game_result = board.get_game_result()

        if game_result == Result.Draw:
            self.draws += 1
            return 0

        if game_result == Result.X_Wins:
            result = 1 if self.turn == 1 else -1
        elif game_result == Result.O_Wins:
            result = 1 if self.turn == 2 else -1

        if result == 1:
            self.wins += 1
        else:
            self.losses += 1

        return result

    def load(self, path):
        loaded_checkpoint = torch.load(path)

        self.online_net.load_state_dict(loaded_checkpoint[Key.Net.name])
        self.online_net.train()

        self.target_net.load_state_dict(loaded_checkpoint[Key.Net.name])
        self.target_net.eval()

        self.optimizer.load_state_dict(loaded_checkpoint[Key.Optimizer.name])

        self.games = loaded_checkpoint[Key.Games.name]
        self.wins = loaded_checkpoint[Key.Wins.name]
        self.draws = loaded_checkpoint[Key.Draws.name]
        self.losses = loaded_checkpoint[Key.Losses.name]
