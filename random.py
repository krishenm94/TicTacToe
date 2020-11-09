from player import Player
from board import Board

import random

class Random(Player):
    """docstring for Random"""
    def __init__(self):
        super(Random, self).__init__()

    def get_best_move(self, board):
        return random.choice(board.get_valid_moves())
        