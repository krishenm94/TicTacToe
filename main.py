from random_player import Random
from minimax import Minimax
from play_game import play_game, play_games
from q_learning import QLearning
from human import Human

# play_games(1000, Minimax(True), Minimax())
# play_games(1000, Minimax(), Minimax(True))
# play_game(QLearning(1,2), Human())
# play_game(QLearning(1), Human())
# play_games(1000, QLearning(1), QLearning(2,True))
# play_games(1000, QLearning(1,True), QLearning(2))
# play_games(1000, QLearning(1), Minimax())
# play_games(1000, Minimax(), QLearning(2))
# play_games(1000, Minimax(), QLearning(2,True))
# play_games(1000, QLearning(1, True), Minimax())
play_games(1000, QLearning(1, False, True), QLearning(2))
play_games(1000, QLearning(1), QLearning(2, False, True))
# play_game(Minimax(), Human())
# play_game(Human(), Minimax())
# play_games(10000, Random() ,Random())
# play_games(1000, Minimax(), Random())
# play_games(1000, Random(), Minimax())
# play_games(10000, Minimax(), Minimax())
