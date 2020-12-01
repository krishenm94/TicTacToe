from player import Player
from cache import Cache1
from board import Cell
from math import inf
import time


class ABPruning(Player):
    """docstring for ABPruning"""

    def __init__(self):
        super(ABPruning, self).__init__("ABPruning")
        self.cache = Cache1()
        self.time_taken = 0

    def get_best_move(self, board):
        t0 = time.time()
        move_value_pairs = self.get_move_values(board)
        t1 = time.time() - t0
        self.time_taken += t1
        print(f"AB time taken: {self.time_taken} s")
        return self.filter(board, move_value_pairs)

    def filter(self, board, move_value_pairs):
        min_or_max = self.min_or_max(board)
        move, value = min_or_max(move_value_pairs, key=lambda pair: pair[1])
        return move

    def get_move_values(self, board):
        moves = board.get_valid_moves()
        assert moves, "No valid moves"

        return [(move, self.get_move_value(move, board, -inf, inf))
                for move in moves]

    def get_move_value(self, move, board, alpha, beta):
        new_board = board.simulate_turn(move)
        cached, found = self.cache.get(new_board)

        if found:
            return cached

        value = self.calculate_position_value(new_board, alpha, beta)
        self.cache.set(new_board, value)
        return value

    def calculate_position_value(self, board, alpha, beta):
        if board.is_game_over():
            return board.get_game_result()

        moves = board.get_valid_moves()

        min_or_max = self.min_or_max(board)

        value = self.get_move_value(moves[0], board, alpha, beta)
        for move in moves:
            value = min_or_max(value, self.get_move_value(move, board, alpha, beta))
            if min_or_max is max:
                alpha = max(alpha, value)
                if alpha >= beta:
                    return value

            else:
                beta = min(beta, value)
                if beta <= alpha:
                    return value

        return value

    def min_or_max(self, board):
        return min if board.whose_turn() == Cell.O else max
