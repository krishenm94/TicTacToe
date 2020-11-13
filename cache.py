from board import Board
import numpy as np

class Cache(object):
    """docstring for Cache"""

    def __init__(self):
        super(Cache, self).__init__()

        self.boards = {}

    def add(self, board, value):
        cells_2d = board.cells_2d
        result = value

        self.boards[cells_2d.tobytes()] = result

        rot90 = np.rot90(cells_2d)
        self.boards[rot90.tobytes()] = result

        rot180 = np.rot90(rot90)
        self.boards[rot180.tobytes()] = result

        rot270 = np.rot90(rot180)
        self.boards[rot270.tobytes()] = result

        self.boards[np.fliplr(cells_2d).tobytes()] = result
        self.boards[np.flipud(cells_2d).tobytes()] = result
        self.boards[np.fliplr(rot90).tobytes()] = result
        self.boards[np.flipud(rot90).tobytes()] = result

    def get(self, board):
        result = self.boards.get(board.cells_2d.tobytes(), None)

        if result is None:
            return None, False

        return result, True
