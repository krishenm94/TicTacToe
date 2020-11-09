from board import Board
from board import Result
from random import Random
from human import Human
from minimax import Minimax

import itertools

def play_game(x_player, o_player):
    x_player.set_turn(1)
    o_player.set_turn(2)
    board = Board()

    players = itertools.cycle([x_player, o_player])

    while not board.is_game_over():
        player = next(players)
        player.move(board)

    return board


def play_games(total_games, x_player, o_player):
    results = {
        Result.X_Wins: 0,
        Result.O_Wins: 0,
        Result.Draw: 0
    }

    for g in range(total_games):
        end_of_game = (play_game(x_player, o_player))
        result = end_of_game.get_game_result()
        results[result] += 1

    x_wins_percent = results[Result.X_Wins] / total_games * 100
    o_wins_percent = results[Result.O_Wins] / total_games * 100
    draw_percent = results[Result.Draw] / total_games * 100

    print(f"x wins: {x_wins_percent:.2f}%")
    print(f"o wins: {o_wins_percent:.2f}%")
    print(f"draw  : {draw_percent:.2f}%")