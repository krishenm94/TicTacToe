from board import Board

class Player(object):
    """docstring for Player"""
    def __init__(self, arg):
        super(Player, self).__init__()
    
    def move(self, board)
        valid_moves = board.get_valid_moves()
        
        assert(valid_moves.size() > 0, "No valid moves.")

        board.execute_turn(get_best_move(), valid_moves);

    def get_best_move(self, moves)
        raise NotImplementedError
