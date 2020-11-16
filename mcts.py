from player import Player
from cache import Cache1

from math import sqrt, inf, log

SQUARE_ROOT_2 = sqrt(2)


class Mcts(Player):
    """docstring for Mcts"""

    class Node(object):
        """docstring for Node"""

        def __init__(self):
            super(Mcts.Node, self).__init__()
            self.wins = 0
            self.visits = 0
            self.parents = Cache1()

        def upper_confidence_bound(self):
            if self.visits == 0:
                return inf

            exploitation_component = self.wins / self.visits
            parent_visits = self.parent_visits()
            exploration_component = SQUARE_ROOT_2 * sqrt(log(parent_visits) / self.visits)

            return exploitation_component + exploration_component

        def parent_visits(self):
            return sum(parent.visits for parent in self.parents.boards.values())

        def add_parent(self, parent_board, parent_node):
            node, found = self.parents.get(parent_board)
            if found:
                return
            # assert not found, "Given parent is already set."

            self.parents.set(parent_board, parent_node)

    def __init__(self, use_depth_quotient=False):
        super(Mcts, self).__init__("Monte Carlo Tree Search", use_depth_quotient)
        self.nodes = Cache1()

    def get_best_move(self, board):
        current_node = self.get_node(board)
        move_child_node_pairs = self.get_move_child_node_pairs(board)

        best_move, best_node = move_child_node_pairs[0]
        for move, node in move_child_node_pairs:
            node.add_parent(current_node)

            if node.upper_confidence_bound() > best_node.upper_confidence_bound():
                best_move, best_node = move, node

        # Where to update child node?
        return best_move

    def get_move_child_node_pairs(self, board):
        return [(move, self.get_child_node(move, board)) for move in board.get_valid_moves()]

    def get_child_node(self, move, board):
        new_board = board.simulate_turn(move)
        cached_node, found = self.nodes.get(new_board)

        if found:
            return cached_node

        return Mcts.Node()
