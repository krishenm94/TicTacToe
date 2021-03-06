from player import Player

import random


class Random(Player):
    """docstring for Random"""

    def __init__(self):
        super(Random, self).__init__("Random")

    def get_best_move(self, board):
        return random.choice(board.get_valid_moves())
