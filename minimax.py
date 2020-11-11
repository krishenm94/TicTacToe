from player import Player
from cache import Cache


class Minimax(Player):
    """docstring for Minimax"""

    def __init__(self):
        super(Minimax, self).__init__()
        self.counter = 0
        self.cache = Cache()

    def get_best_move(self, board):
        move_value_pairs = self.get_move_values(board)
        return self.filter(move_value_pairs)

    def filter(self, move_value_pairs):
        move, value = max(move_value_pairs,
                          key=lambda pair: pair[1] * -1 ** (self.turn - 1))

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
            new_board.print()
            print("Cached value, depth: %f, %f" % ((cached), new_board.get_depth()))
            return cached

        # self.counter += 1
        # print("Counter: %d, Upper Limit: 9! = 362880" %self.counter)

        if (new_board.is_game_over()):
            new_board.print()
            print("Simulation over, move score: %f" % (new_board.get_game_result() / new_board.get_depth()))
            return  new_board.get_game_result() / new_board.get_depth()

        value = self.calculate_position_value(new_board)
        self.cache.add(board, value/new_board.get_depth())

        return value

    def calculate_position_value(self, board):
        moves = board.get_valid_moves()

        move_values = [-1 ** (board.get_depth() + self.turn - 1) *
                       self.get_move_value(move, board)
                       for move in moves]

        return max(move_values)
