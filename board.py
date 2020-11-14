from enum import IntEnum
import numpy as np


class Cell(IntEnum):
    Empty = 0
    O = -1  # player 2
    X = 1  # player 1


class Result(IntEnum):
    X_Wins = 1
    O_Wins = -1
    Draw = 0
    Incomplete = 2


SIZE = 3


class Board(object):
    """docstring for Board"""

    def __init__(self, cells=None, illegalMove=None):
        super(Board, self).__init__()
        if cells is None:
            self.cells = np.array([Cell.Empty] * SIZE ** 2)
        else:
            self.cells = cells.copy()

    def cells_2d(self):
        return self.cells.reshape(SIZE, SIZE)

    def execute_turn(self, move):
        assert self.cells[move] == Cell.Empty, "Cell is not empty"

        self.cells[move] = self.whose_turn()
        return

    def whose_turn(self):
        non_zero_count = np.count_nonzero(self.cells)
        return Cell.X if (non_zero_count % 2 == 0) else Cell.O

    def get_valid_moves(self):
        return [i for i in range(self.cells.size)
                if self.cells[i] == Cell.Empty]

    def simulate_turn(self, move):
        new_board = Board(self.cells)
        new_board.execute_turn(move)
        return new_board

    def print(self):
        rows, cols = self.cells_2d().shape
        print('\n')

        for row in range(rows):
            print('|', end="")

            for col in range(cols):
                cell = self.cells_2d()[row][col]
                print(" %s " % self.cell_to_char(cell), end="|")

            if row < rows - 1:
                print("\n-------------")

        print('\n')

    def cell_to_char(self, cell):

        if cell == Cell.Empty:
            return ' '

        if cell == Cell.X:
            return 'X'

        if cell == Cell.O:
            return 'O'

        assert (False, "Undefined tic tac toe cell")

    def is_move_valid(self, move):
        if move > (SIZE ** 2 - 1):
            return False

        if self.cells[move] == Cell.Empty:
            return True

        return False

    def is_game_over(self):
        return self.get_game_result() != Result.Incomplete

    def get_game_result(self):
        rows_cols_and_diagonals = self.get_rows_cols_and_diagonals()

        sums = list(map(sum, rows_cols_and_diagonals))
        max_value = max(sums)
        min_value = min(sums)

        if max_value == SIZE:
            return Result.X_Wins

        if min_value == -SIZE:
            return Result.O_Wins

        if not self.get_valid_moves():
            return Result.Draw

        return Result.Incomplete

    def get_rows_cols_and_diagonals(self):
        rows_and_diagonal = self.get_rows_and_diagonal(self.cells_2d())
        cols_and_antidiagonal = self.get_rows_and_diagonal(np.rot90(self.cells_2d()))
        return rows_and_diagonal + cols_and_antidiagonal

    def get_rows_and_diagonal(self, cells_2d):
        num_rows = cells_2d.shape[0]
        return ([row for row in cells_2d[range(num_rows), :]]
                + [cells_2d.diagonal()])

    def get_depth(self):
        return sum(cell != Cell.Empty for cell in self.cells)
