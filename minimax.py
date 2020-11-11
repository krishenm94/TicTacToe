from player import Player

class Minimax(Player):
    """docstring for Minimax"""
    def __init__(self):
        super(Minimax, self).__init__()
        self.counter = 0
 
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

        return [(move, self.get_move_value(move, 0, board)) 
                for move in moves]

    def get_move_value(self, move, depth, board):
        new_board = board.simulate_turn(move)
        depth += 1

        self.counter += 1
        print("Counter: %d, Upper Limit: 9! = 362880" %self.counter)

        if (new_board.is_game_over()):
            new_board.print()
            print("Simulation over")
            return new_board.get_game_result()

        return self.calculate_position_value(depth, new_board)

    def calculate_position_value(self, depth, board):

        moves = board.get_valid_moves()

        move_values = [-1 ** (depth + self.turn - 1) *
                       self.get_move_value(move, depth, board)
                       for move in moves]

        return max(move_values)
