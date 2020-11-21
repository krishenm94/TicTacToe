from random_player import Random
from minimax import Minimax
from play_game import play_game, play_games
from qlearning import QLearning
from mcts import Mcts
from human import Human

tree = Mcts()
minimax = Minimax()
random = Random()
tree.train()
# tree.debug = True
# play_game(tree, Human())
# play_game(Human(), tree)
play_games(1000, tree, random)
play_games(1000, random, tree)
play_games(1000, tree, minimax)
play_games(1000, minimax, tree)
play_games(1000, tree, tree)

