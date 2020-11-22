from random_player import Random
from minimax import Minimax
from play_game import play_game, play_games
from qlearning import QLearning
from mcts import Mcts
from human import Human
from qneural import QNeural

from torch.nn import MSELoss

human = Human()
tree = Mcts()
minimax = Minimax()
random = Random()
# tree.train()

x_learning = QLearning()
o_learning = QLearning()


neural = QNeural(MSELoss())

neural.train(1)
play_games(1000, neural, random)
play_games(1000, neural, minimax)

neural.games = 0
neural.train(2)

play_games(1000, neural, random)
play_games(1000, random, neural)


play_games(1000, neural, minimax)
play_games(1000, minimax, neural)
