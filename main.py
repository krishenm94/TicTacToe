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

x_neural = QNeural(MSELoss())
# o_neural = QNeural(MSELoss())

x_neural.train(1)

play_games(1000, x_neural, random)
# play_games(1000, random, o_neural)
