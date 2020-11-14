from player import Player
from cache import Cache
from board import Board

import numpy as np
import random
import operator

INITIAL_Q_VALUE = 0


class QTable(object):
    """docstring for QTable"""

    def __init__(self):
        super(QTable, self).__init__("QTable")
        self.cache = Cache()

    def get_values(self, board):
        moves = board.get_valid_moves()
        q_values = [self.get_value(board, move) for move in moves]

        return dict(zip(moves, q_values))

    def get_value(self, board, move):
        new_board = board.simulate_turn(move)
        cached, found = self.cache.get(new_board)
        if found is True:
            return cached

        return INITIAL_Q_VALUE

    def update_value(self, board, move, value):
        new_board = board.simulate_turn(move)

        self.cache.set(new_board, value)

    def get_max_value_and_its_move(self, board):
        return max(self.get_values(board).items(), key=operator.itemgetter(1))

    def print(self):
        print(f"num q_values = {len(self.cache.boards)}")
        for cells_bytes, value in self.cache.boards.items():
            cells = np.frombuffer(cells_bytes, dtype=int)
            board = Board(cells)
            board.print()
            print(f"qvalue = {value}")

class QLearning(Player):
    """docstring for QLearning"""

    def __init__(self):
        super(QLearning, self).__init__("QLearning")
        self.table = QTable()

    def choose_move_index(self, board, epsilon):
        if epsilon > 0:
            random_value_from_0_to_1 = np.random.uniform()
            if random_value_from_0_to_1 < epsilon:
                return random.choice(board.get_valid_moves())

        move_q_value_pairs = self.get_move_average_q_value_pairs(board)

        return max(move_q_value_pairs, key=lambda pair: pair[1])[0]

    def get_move_average_q_value_pairs(self, board):
        return
