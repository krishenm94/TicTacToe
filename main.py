from random_player import Random
from minimax import Minimax
from play_game import play_game, play_games
from q_learning import QLearning

play_games(QLearning(1), Random(), 1000)
# play_game(Minimax(), Human())
# play_game(Human(), Minimax())
# play_games(10000, Random() ,Random())
# play_games(10000, Minimax(), Random())
# play_games(10000, Random(), Minimax())
# play_games(10000, Minimax(), Minimax())
