from enum import IntEnum
import numpy as np 

class Cell(IntEnum): 
    Empty = 0
    O = 2 # player 2 
    X =  1 # player 1

class Result(IntEnum):
    WIN_X = 2
    WIN_O = 0
    DRAW = 1
    INCOMPLETE = 3

SIZE = 3

class Board(object):
    """docstring for Board"""
    def __init__(self, cells = None, illegalMove = None):
        super(Board, self).__init__()
        if cells is None:
            self.cells = np.array(Cell.Empty * SIZE **2)
        else:
            self.cells = cells;

        self.cells_2d = self.cells.reshape(SIZE, SIZE) 

    def execute_turn(self, move)
        assert(self.cells[move] == Cell.Empty, "Cell is not empty")
            
        self.cells[move] = whose_turn()
        return

    def whose_turn(self):
        non_zero = np.count_nonzero(self.cells)
        return Cell.X if is_even(non_zero) else Cell.O

    def get_valid_moves(self)
        return [i for i in range(self.cells.size) 
                if self.board[i] == Cell.Empty]

    def simulate_turn(self, move)
        new_board = Board(self.cells)
        new_board.execute_turn(move)
        return new_board;
    
    def print(self)
        rows, cols = self.cells_2d.shape
        print('\n')

        for row in range(rows)
            print('|')
            
            for col in range(cols)
                cell = self.cells_2d[row][col]
                print(" %s |" % self.cell_to_char(cell))
            
            if (row < rows - 1)
                print("-------------")

        print('\n')

    def cell_to_char(self, cell)

        if (cell == Cell.Empty)
            return ' '
        else if (cell == Cell.X)
            return 'X'
        else if (cell == Cell.O)
            return 'O'              
        
        assert(false, "Undefined tic tac toe cell")

    def is_move_valid(self, move)
        if (move > SIZE ** 2 -1)
            return False

        if (self.cells[move] == Cell.Empty)
            return True
        else
            return false
