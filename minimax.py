from player import Player
from cache import Cache1
from cache import Cache2
from board import Cell


class Minimax(Player):
    """docstring for Minimax"""

    def __init__(self):
        super(Minimax, self).__init__("Minimax")
        self.cache = Cache1()
        # self.cache = Cache2()

    def get_best_move(self, board):
        move_value_pairs = self.get_move_values(board)
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
            # print("Board found in cache, value: %f" %cached)
            # new_board.print()
            return cached

        value = self.calculate_position_value(new_board)
        self.cache.set(new_board, value)

        return value

    def calculate_position_value(self, board):
        if board.is_game_over():
            return board.get_game_result()

        moves = board.get_valid_moves()

        min_or_max = self.min_or_max(board)

        move_values = [self.get_move_value(move, board)
                       for move in moves]

        return min_or_max(move_values)

    def min_or_max(self, board):
        return min if board.whose_turn() == Cell.O else max
