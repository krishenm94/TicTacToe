from player import Player

import torch
from torch import nn

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


class NeuralModel(object):
    def __init__(self, policy_net, target_net, optimizer, loss_function):
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optimizer
        self.loss_function = loss_function


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
        output = get_target_output()

