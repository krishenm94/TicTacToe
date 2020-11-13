from player import Player
from cache import Cache
import numpy as np

class QLearning(Player):
    """docstring for QLearning"""

    def __init__(self):
        super(QLearning, self).__init__("QLearning")
        self.table = Cache()

    def get_best_move(self, board):
        return self.choose_move_index(board, 0)

    def choose_move_index(self, board, epsilon):
        if epsilon > 0:
            random_value_from_0_to_1 = np.random.uniform()
            if random_value_from_0_to_1 < epsilon:
                return board.get_random_valid_move()

        move_q_value_pairs = self.get_move_average_q_value_pairs(board)

        return max(move_q_value_pairs, key=lambda pair: pair[1])[0]



