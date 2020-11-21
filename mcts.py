from player import Player
from cache import Cache1
from board import Board, Result, Cell

from math import sqrt, inf, log
from tqdm import trange
from time import sleep

SQUARE_ROOT_2 = sqrt(2)
PLAYOUTS = 5000


class Mcts(Player):
    """docstring for Mcts"""

    class Node(object):
        """docstring for Node"""

        def __init__(self):
            super(Mcts.Node, self).__init__()
            self.wins = 0
            self.draws = 0
            self.losses = 0
            self.visits = 0
            self.parents = Cache1()

        def upper_confidence_bound(self):
            if self.visits == 0:
                return inf

            exploitation_component = (self.wins + self.draws) / self.visits
            parent_visits = self.parent_visits()
            exploration_component = SQUARE_ROOT_2 * sqrt(log(parent_visits) / self.visits)

            return exploitation_component + exploration_component

        def parent_visits(self):
            return sum(parent.visits for parent in self.parents.boards.values())

        def register_parent(self, parent_board, parent_node):
            node, found = self.parents.get(parent_board)
            if found:
                return

            self.parents.set(parent_board, parent_node)

    def __init__(self, use_depth_quotient=False):
        super(Mcts, self).__init__("Monte Carlo Tree Search", use_depth_quotient)
        self.nodes = Cache1()

    def get_best_move(self, board):
        current_node = self.get_node(board)
        move_child_node_pairs = self.get_move_child_node_pairs(board)

        # Forward propagation, create tree structure
        best_move, best_node = move_child_node_pairs[0]
        for move, node in move_child_node_pairs:
            node.register_parent(current_node)

            if node.upper_confidence_bound() > best_node.upper_confidence_bound():
                best_move, best_node = move, node

        return best_move

    def get_node(self, board):
        return self.nodes.get(board)

    def get_move_child_node_pairs(self, board):
        return [(move, self.get_child_node(move, board))
                for move in board.get_valid_moves()]

    def get_child_node(self, move, board):
        new_board = board.simulate_turn(move)
        cached_node, found = self.get_node(new_board)

        if found:
            return cached_node

        return Mcts.Node()

    def train(self, board=Board(), playouts=PLAYOUTS):
        print(f"Performing {playouts} playouts.")

        sleep(0.05)
        for _ in trange(playouts):
            self.playout(board)

    def playout(self, board):
        history = [board]

        while not board.is_game_over():
            move = self.get_best_move(board)
            board = board.simulate_turn(move)
            history.append(board)

        result = board.get_game_result()
        self.backpropagate(history, result)

    def backpropagate(self, history, game_result):
        for board in history:
            node = self.get_node(board)
            node.visits += 1
            if self.is_win(board.whose_turn(), game_result):
                node.wins += 1
            elif self.is_loss(board.whose_turn(), game_result):
                node.losses += 1
            elif game_result == Result.Draw:
                node.draws += 1
            else:
                raise ValueError("Illegal game state.")

    def is_win(self, turn, result):
        return turn == Cell.X and result == Result.O_Wins or \
               turn == Cell.O and result == Result.X_Wins

    def is_loss(self, turn, result):
        return turn == Cell.X and result == Result.X_Wins or \
               turn == Cell.O and result == Result.O_Wins
