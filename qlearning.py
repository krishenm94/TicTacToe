from player import Player
from cache import Cache1, Cache2
from board import Board, Result, Cell
from random_player import Random
from minimax import Minimax

import numpy as np
import random
import operator
import statistics as stats
from collections import deque
import itertools
from tqdm.auto import trange, tqdm
import time

INITIAL_Q_VALUE = 0
TOTAL_GAMES = 2000


class QLearning(Player):
    """docstring for QLearning"""

    class Table(object):
        """docstring for QTable"""

        def __init__(self):
            super(QLearning.Table, self).__init__()
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

    def __init__(self, use_double=False):
        super(QLearning, self).__init__("QLearning")

        self.tables = [QLearning.Table()]
        if use_double:
            self.tables.append(QLearning.Table())
            self.name = "Double " + self.name

        self.learning_rate = 0.4
        self.discount_factor = 1.0
        self.initial_epsilon = 0.7
        self.move_history = deque()

    def get_best_move(self, board):
        return self.choose_move_index(board, 0)

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

    def gather_values_for_move(self, board, move):
        return [table.get_value(board, move) for table in self.tables]

    def train(self, turn, opponent=Random(), total_games=TOTAL_GAMES):
        print(f"Training {self.name} for {total_games} games.", flush=True)
        self.turn = turn
        opponent.set_turn(self.turn % 2 + 1)
        epsilon = self.initial_epsilon

        time.sleep(0.05)  # Ensures no collisions between tqdm prints and main prints
        for game in trange(total_games):

            self.play_training_game(opponent, epsilon)
            # Decrease exploration probability
            if (game + 1) % (total_games / 10) == 0:
                epsilon = max(0, epsilon - 0.1)
                # tqdm.write(f"{game + 1}/{total_games} games, using epsilon={epsilon}...")

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

    def post_training_game_update(self, board, move_history):
        end_state_value = self.get_end_state_value(board)

        # Initialize tables
        # Update occurs reverse chronologically
        next_board, move = move_history[0]
        for table in self.tables:
            current_value = table.get_value(next_board, move)
            new_value = self.calculate_new_value(current_value, end_state_value, 0)
            table.update_value(next_board, move, new_value)

        # Complete learning
        for board, move in list(move_history)[1:]:
            current_table, next_table = self.get_shuffled_tables()

            next_move, _ = current_table.get_max_value_and_its_move(next_board)
            max_next_value = next_table.get_value(next_board, next_move)
            current_value = current_table.get_value(board, move)
            new_value = self.calculate_new_value(current_value, 0, max_next_value)

            current_table.update_value(board, move, new_value)

            next_board = board

    def get_shuffled_tables(self):
        tables = self.tables.copy()
        random.shuffle(tables)
        table_cycle = itertools.cycle(tables)

        current_table = next(table_cycle)
        next_table = next(table_cycle)

        return current_table, next_table

    def get_end_state_value(self, board):
        assert board.is_game_over(), "Game is not over"

        game_result = board.get_game_result()

        if game_result == Result.Draw:
            return 1

        if game_result == Result.X_Wins:
            return 1 if self.turn == 1 else -1

        if game_result == Result.O_Wins:
            return 1 if self.turn == 2 else -1

        assert False, "Undefined behaviour"

    def calculate_new_value(self, current_value, reward, max_next_value):
        prior_component = (1 - self.learning_rate) * current_value
        next_component = self.learning_rate * (reward + self.discount_factor * max_next_value)
        return prior_component + next_component
