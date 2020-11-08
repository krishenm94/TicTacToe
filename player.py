from board import Board

class Player(object):
    """docstring for Player"""
    def __init__(self):
        super(Player, self).__init__()
    
    def move(self, board)
        valid_moves = board.get_valid_moves()
        
        assert(valid_moves.size() > 0, "No valid moves.")

        board.execute_turn(self.get_best_move(board));

    def get_best_move(self, board)
        raise NotImplementedError
