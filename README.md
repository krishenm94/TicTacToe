# TicTacToe
Exploring different machine learning solutions to tic tac toe

References:
1. https://nestedsoftware.com/2019/06/15/tic-tac-toe-with-the-minimax-algorithm-5988.123625.html
2. https://becominghuman.ai/minimax-or-maximin-8772fbd6d0c2

## Algorithms

1. Minimax
2. Tabular Q-Learning
3. Monte Carlo Tree Search
4. Neural
5. Minimax with alpha-beta pruning

## Results:

With depth quotient refers to the reward function with a quotient equivalent to the number of turns taken. This way the algorithm is incentived to end the game sooner rather than later.

### Minimax

#### Results

// With Depth Quotient and Cache1

Playing 10000 games
x wins: 98.59%
o wins: 0.94%
draw  : 0.47%

Random as X and Minimax as O
Playing 10000 games
x wins: 21.42%
o wins: 69.15%
draw  : 9.43%

// With Depth Quotient and Cache2

Minimax as X and Random as O
Playing 10000 games
x wins: 98.56%
o wins: 0.94%
draw  : 0.50%

Random as X and Minimax as O
Playing 10000 games
x wins: 24.62%
o wins: 67.64%
draw  : 7.74%

// Without Depth Quotient and Cache1

Minimax as X and Random as O
Playing 10000 games
x wins: 95.13%
o wins: 3.37%
draw  : 1.50%

Random as X and Minimax as O
Playing 10000 games
x wins: 20.39%
o wins: 66.54%
draw  : 13.07%

// Without Depth Quotient and Cache2

Minimax as X and Random as O
Playing 10000 games
x wins: 94.42%
o wins: 3.75%
draw  : 1.83%

Random as X and Minimax as O
Playing 10000 games
x wins: 17.04%
o wins: 68.41%
draw  : 14.55%

### Q-Learning

Q-learning here was implemented with epsilon-greedy action selection with an epsilon tending towards 0 as t tends towards infinite. This was done to promote exploration over exploitation initially.