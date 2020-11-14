from player import Player
from cache import Cache1
from board import Board
from random_player import Random
from minimax import Minimax

import numpy as np
import random
import operator
import statistics as stats
from collections import deque

INITIAL_Q_VALUE = 0


class QTable(object):
    """docstring for QTable"""

    def __init__(self):
        super(QTable, self).__init__("QTable")
        self.cache = Cache1()

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
        self.tables = [QTable()]
        # self.tables = [QTable(), QTable()]
        self.learning_rate = 0.4
        self.discount_factor = 1.0
        self.initial_epsilon = 0.7

    def choose_move_index(self, board, epsilon):
        if epsilon > 0:
            random_value_from_0_to_1 = np.random.uniform()
            if random_value_from_0_to_1 < epsilon:
                return random.choice(board.get_valid_moves())

        move_value_pairs = self.get_move_average_value_pairs(board)

        return max(move_value_pairs, key=lambda pair: pair[1])[0]

    def get_move_average_value_pairs(self, board):
        moves = sorted(self.tables[0].get_values(board).keys())

        mean_values = [stats.mean(self.gather_values_for_move(board, move))
                       for move in moves]

        return list(zip(moves, mean_values))

    def train(self, opponent=Random(), total_games=5000):

        opponent.set_turn(self.turn % 2 + 1)
        epsilon = self.initial_epsilon

        for game in range(total_games):
            move_history = deque()

            self.play_training_game(move_history, opponent, epsilon)

            # Decrease exploration probability
            if (game + 1) % (total_games / 10) == 0:
                epsilon = max(0, epsilon - 0.1)
                print(f"{game + 1}/{total_games} games, using epsilon={epsilon}...")

    def play_training_game(move_history, opponent, epsilon):
