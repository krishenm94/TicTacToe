from board import Board

class Cache(object):
    def __init__(self):
        super(Cache, self).__init__()
        self.boards = list();

    def add(self, board):
        