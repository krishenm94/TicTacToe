# TicTacToe
Exploring different machine learning solutions to tic tac toe

References:
1. https://nestedsoftware.com/2019/06/15/tic-tac-toe-with-the-minimax-algorithm-5988.123625.html
2. https://becominghuman.ai/minimax-or-maximin-8772fbd6d0c2

## Algorithms

1. Minimax
2. qLearning
3. Monte Carlo Tree Search
4. Neural
5. Minimax with alpha-beta pruning



## Results:

With depth quotient refers to the reward function with a quotient equivalent to the number of turns taken. This way the algorithm is incentived to end the game sooner rather than later.

### Minimax

// With Depth Quotient

Minimax as X and Random as O
Playing 5000 games
x wins: 98.38%
o wins: 1.12%
draw  : 0.50%
Random as X and Minimax as O
Playing 5000 games
x wins: 13.22%
o wins: 77.36%
draw  : 9.42%

// Without

Minimax as X and Random as O
Playing 5000 games
x wins: 94.28%
o wins: 3.80%
draw  : 1.92%
Random as X and Minimax as O
Playing 5000 games
x wins: 20.40%
o wins: 66.12%
draw  : 13.48%