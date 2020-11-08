#!/usr/bin/python3

from enum import IntEnum
import numpy as np

class Cell(IntEnum): 
    EMPTY = 0
    O = 1
    X =  2

class Result(IntEnum):
    WIN_X = 0
        DRAW = 2
    WIN_O = 1

    INCOMPLETE = 3


class Board(object):
    """docstring for Board"""
    def __init__(self, size = 3, cells = None, illegalMove = None):
        super(Board, self).__init__()
        if board is None:
            self.cells = np.array(Cell.Empty * size **2)
        else:
            self.cells = cells;

        self.board = self.cells.reshape(size, size) 

    def execute_turn(self, move)
        if(is_valid(move)):
            self.cells[move] = whose_turn()
            return True 
        else:
            return False

    def whose_turn(self):
        non_zero = np.count_nonzero(self.cells)
        return Cell.X if is_even(non_zero) else Cell.O
            
    def is_valid(move)
        if(self.cells[move] == Cell.EMPTY):
            return True
        else:
            return False

    def get_valid_moves(self)
        return [i for i in range(self.cells.size) 
                if self.board[i] == Cell.EMPTY]
                    