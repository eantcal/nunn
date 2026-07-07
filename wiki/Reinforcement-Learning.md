# Reinforcement Learning

Reinforcement learning models interaction as a loop between an agent and an environment.

![Agent environment loop](assets/rl-agent-environment.png)

At each step:

1. the agent observes a state;
2. it chooses an action;
3. the environment returns a new state and a reward;
4. the agent updates its policy or value estimates.

## Q-learning

Q-learning is an off-policy temporal-difference method. It learns an estimate:

```text
Q(s, a)
```

for the expected return of taking action `a` in state `s`.

The update is based on the best next action, regardless of the action actually selected by the exploration policy:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

## SARSA

SARSA is on-policy. It updates using the action actually selected at the next state:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
```

This makes it more conservative when exploration can be risky.

## Maze Demos

![Maze rewards](assets/maze-rewards.png)

The maze examples show how reward shaping guides the agent toward the goal while discouraging invalid or undesirable moves.

![Q-table](assets/q-table.png)

Demo programs:

- `maze`
- `path_finder`

## DQN

`Dqn` replaces a tabular Q-table with an `MlpMatrixNN`. It includes:

- replay buffer;
- random mini-batch sampling;
- frozen target network;
- epsilon-greedy action selection.

Demo:

- `dqn_maze`

