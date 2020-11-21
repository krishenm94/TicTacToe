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
x_learning = QLearning(1)
o_learning = QLearning(2)
# play_game(tree, Human())
# play_game(Human(), tree)
# play_games(1000, tree, random)
# play_games(1000, random, tree)
# play_games(1000, tree, minimax)
# play_games(1000, minimax, tree)
# play_games(1000, tree, tree)
play_games(1000, QLearning(1), tree)
play_games(1000, tree, QLearning(2))

