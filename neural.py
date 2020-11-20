from player import Player
from board import Board, Cell, Result

import torch
from torch import nn
import numpy as np

class Net(nn.Module):
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
        output = self.get_target_output(board)
        valid_output_move_pairs = self.filter_output(output, board)
        best_move, _ = max(valid_output_move_pairs, key=lambda pair : pair[1])
        return best_move

    def get_target_output(self, board):
        input = torch.tensor(board.cells, dtype=torch.float)
        return self.target_net(input)

    def filter_output(self, output, board):
        valid_moves = board.get_valid_moves()
        valid_output_move_pairs = []
        for move in valid_moves:
            valid_output_move_pairs.append((move, output[move].item()))
        return valid_output_move_pairs
    



