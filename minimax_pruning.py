from player import Player
from cache import Cache1
from board import Cell
from math import inf
import time


class MinimaxABPruning(Player):
    """docstring for MinimaxABPruning"""

    def __init__(self):
        super(MinimaxABPruning, self).__init__("MinimaxABPruning")
        self.cache = Cache1()
        self.alpha = -inf
        self.beta = inf

    def get_best_move(self, board):
        t0 = time.time()
        move_value_pairs = self.get_move_values(board)
        t1 = time.time() - t0
        print(f"AB Time taken: {t1}")
        return self.filter(board, move_value_pairs)

    def filter(self, board, move_value_pairs):
        min_or_max = self.min_or_max(board)
        move, value = min_or_max(move_value_pairs, key=lambda pair: pair[1])
        return move

    def get_move_values(self, board):
        moves = board.get_valid_moves()
        assert moves, "No valid moves"

        return [(move, self.get_move_value(move, board))
                for move in moves]

    def get_move_value(self, move, board):
        new_board = board.simulate_turn(move)
        cached, found = self.cache.get(new_board)

        if found:
            return cached

        value = self.calculate_position_value(new_board)
        self.cache.set(new_board, value)
        return value

    def calculate_position_value(self, board):
        if board.is_game_over():
            return board.get_game_result()

        moves = board.get_valid_moves()

        min_or_max = self.min_or_max(board)

        for move in moves:
            value = self.get_move_value(move, board)
            if min_or_max is max:
                self.alpha = max(self.alpha, value)

                if self.alpha >= self.beta:
                    return value
            else:
                self.beta = min(self.beta, value)

                if self.beta <= self.alpha:
                    return value

        return value


    def min_or_max(self, board):
        return min if board.whose_turn() == Cell.O else max
