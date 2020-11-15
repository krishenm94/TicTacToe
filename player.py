from board import Board

class Player(object):
    """docstring for Player"""

    def __init__(self, name, use_depth_quotient=False):
        super(Player, self).__init__()
        self.turn = 0
        self.name = name
        self.use_depth_quotient = use_depth_quotient
        if use_depth_quotient:
            self.name += " + Depth Quotient"

    def set_turn(self, turn):
        assert (self.turn >= 1 or self.turn <= 2,
                "Invalid turn set. Player is designed for 2 player games.")

        self.turn = turn

    def move(self, board):
        assert (self.turn >= 1 or self.turn <= 2,
                "Invalid turn set. Player is designed for 2 player games.")

        board.execute_turn(self.get_best_move(board))

    def get_best_move(self, board):
        raise NotImplementedError
