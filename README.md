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

All algorithms have the option to use a depth quotient for their reward function. This quotient is equivalent to the number of turns taken. This way the algorithm is incentived to end the game sooner rather than later.

### Minimax


### Q-Learning

Q-learning here was implemented with epsilon-greedy action selection with an epsilon tending towards 0 as t tends towards infinite. This was done to promote exploration over exploitation initially.

Q-Learning seems to perform better with Cache1 as opposed to Cache2.