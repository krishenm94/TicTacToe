from board import Board
from player import Player

class Human(Player):
    """docstring for Human"""
    def __init__(self):

        super(Human, self).__init__()

    def get_best_move(self, board):
        move = self.prompt()

        while ( not board.is_move_valid(move)):
            print("Invalid move. Try harder sub-organism.")
            move = self.prompt()

        print("\nYou amuse us.\n")

        return move
        
    def prompt(self):
        return int(input("\nEnter move index human...: \n"))