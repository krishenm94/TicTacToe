# TicTacToe
Exploring different machine learning solutions to tic tac toe

References:
1. https://nestedsoftware.com/2019/06/15/tic-tac-toe-with-the-minimax-algorithm-5988.123625.html
2. https://becominghuman.ai/minimax-or-maximin-8772fbd6d0c2

## Algorithms

1. Minimax
2. Tabular Q-Learning
3. Monte Carlo Tree Search
4. Q Neural Network
5. Minimax with alpha-beta pruning

All algorithms have the option to use a depth quotient for their reward function. This quotient is equivalent to the number of turns taken. This way the algorithm is incentived to end the game sooner rather than later.

### Minimax

Minimax is a decision rule for minimizing the possible loss for a maximum loss scenario. It works by alternating maxing and mining a game score, i.e. changing perspectives between the self and the opponent while walking down the node chain.

Although it is a perfect algorithm for zero sum games, it is computationally expensive as it requires traversing all nodes of the decision tree starting at the node of the current board.


### Q-Learning

Q-learning here was implemented with epsilon-greedy action selection with an epsilon tending towards 0 as t tends towards infinite. This was done to promote exploration over exploitation initially.

Q-Learning seems to perform better with Cache1 (getting all symmetrical states) as opposed to Cache2 (getting all symmetrical states).

To curb the over-optimism or potential positive feedback of Q-Learning greedy (max) policy, the use of two tables can be used. This approach is known as Double Q-Learning. In the Double Q-Learning, the one table is updated based on the best action selected from the other table. Table selection here is done randomly to prevent coupling.

Double Q-Learning seems to require more training to achieve parity with Q-Learning but makes more accurate judgements beyond that.

So far Q-Learning performance has only been evaluated offline. Planning to evaluate online performance as well.

Future work:
1. Exploring online performance
2. Turn independent training

### Monte Carlo Tree Search

Monte Carlo Tree Search is a heuristic decision tree, in more complex problems where computing all possible paths are impossible it is preferred.

UCB (Upper confidence bound) was used to determine the exploitation / exploration ratio, a.k.a. UCT. 

As t tends towards infinite it's behaviour approaches that of minimax.

Future work:
1. Exploring online performance

### Q Neural Network

The design of the network used in this implementation is heavily based on DeepMind's work on Deep Q Networks (DQNs): https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf.

Neural networks applied to Q-Learning allow for a more flexible representation of the state-space.

The two features of DQNs that allow neural networks to be applied in Reinforcement Learning (RL) are:
1. Experience Replay
2. The use of a target and online network

Experience replay was not applied here to simplify the implementation but the use of separate online and target networks was.

An additional augmentation mirroring Double Q-Learning is known as Double DQN. In which the max operation in the target is decomposed into action selection and action evaluation. The online network selects the action and the target network evaluates the action's value. This alteration was also included.

Future work:
1. Exploring online performance
2. Turn independent training

### Minimax with Alpha-Beta Pruning

