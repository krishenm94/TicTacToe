from board import Board
from player import Player

class Minimax(Player):
    """docstring for Minimax"""
    def __init__(self):
        super(Minimax, self).__init__()
    
    def get_best_move(self, board)
        move_value_pairs = self.get_move_values(board)
        return self.filter(move_value_pairs)

    def get_move_values(self, board)
        moves = board.get_valid_moves()
        assert not is_empty(moves), "No moves"

        return [(move, self.get_move_value(move, board)) for move in moves]

    def get_move_value(self, move, board)
        new_board = board.execute_turn(move)

        if (new_board.is_game_over())
            return new_board.game_result()






