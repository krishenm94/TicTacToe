from board import Board

class Player(object):
    """docstring for Player"""
    def __init__(self):
        super(Player, self).__init__()
        self.turn = 0

    def set_turn(turn)
        assert(self.turn >= 1 || self.turn <= 2,
         "Invalid turn set. Player is designed for 2 player games.")
        
        self.turn = turn
    
    def move(self, board)
        assert(self.turn >= 1 || self.turn <= 2,
         "Invalid turn set. Player is designed for 2 player games.")
        
        valid_moves = board.get_valid_moves()
        
        assert(valid_moves.size() > 0, "No valid moves.")

        board.execute_turn(self.get_best_move(board));

    def get_best_move(self, board)
        raise NotImplementedError
