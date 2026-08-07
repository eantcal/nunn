# Reinforcement Learning

Reinforcement learning replaces fixed labels with interaction. An agent chooses an action in a state, observes a reward and next state, then updates an estimate of long-term return. nuNN includes generic tabular Q-learning and SARSA templates plus a concrete DQN built on `MlpMatrixNN`.

![Agent-environment loop](assets/rl-agent-environment.png)

## The contract between environment and learner

The tabular templates do not prescribe a state or action type. Instead, the user-supplied `Agent` exposes the operations the algorithm needs:

```cpp
agent.getCurrentState();
agent.getValidActions();
agent.doAction(action);
agent.reward();
agent.goal();
```

`State` and `Action` must work as keys in the selected map types. The default Q map is an unordered map of state to an unordered map of action to value, so default use also needs equality and hashing.

The complete contract becomes clear in [`maze.cc`](https://github.com/eantcal/nunn/blob/main/examples/maze/maze.cc), where a grid state, four movement actions, an environment, and an agent are all defined beside the learner.

## Policies supplied by nuNN

| Policy | Exploration control | Source |
| --- | --- | --- |
| `EGreedyPolicy<Action, Agent>` | `epsilon` | [`nu_e_greedy_policy.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_e_greedy_policy.h) |
| `SoftmaxPolicy<Action, Agent>` | `temperature` | [`nu_softmax_policy.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_softmax_policy.h) |

The policy object passed to `learn()` selects exploratory actions. `selectAction()` on the trained learner requests the policy's learned/greedy action path.

The current epsilon-greedy implementation also randomizes when the best known value is exactly zero, treating that case as unexplored. This behavior is visible and marked for redesign in the header; account for it when zero is a meaningful learned value.

## Q-learning

Q-learning bootstraps from the best estimated next action:

```text
Q(s,a) <- Q(s,a)
          + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
```

It is off-policy: the update targets a greedy next value even when an exploratory policy collected the transition.

The template assembly used by the maze example is:

```cpp
using Policy = nu::EGreedyPolicy<Action, Agent>;
using Learner = nu::QLearn<Action, State, Agent, Policy>;

Learner learner;
learner.setLearningRate(0.1);
learner.setDiscountRate(0.9);

Agent agent(environment, start, goal);
Policy policy;
policy.setEpsilon(0.1);

double accumulatedUpdateValue = learner.learn(agent, policy);
```

`learn()` runs until `agent.goal()` or an optional listener stops the episode. The return value accumulates the values returned by the internal Q updates; it is not automatically the raw episodic reward. Track environment return separately when that metric matters.

Read the full update, including terminal/dead-end handling, in [`nu_qlearn.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_qlearn.h). Tests are in [`test_rl.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rl.cc).

## SARSA

SARSA bootstraps from the action the current policy actually selects next:

```text
Q(s,a) <- Q(s,a)
          + alpha * (reward + gamma * Q(s',a') - Q(s,a))
```

The public shape is intentionally parallel:

```cpp
using Policy = nu::EGreedyPolicy<Action, Agent>;
using Learner = nu::Sarsa<Action, State, Agent, Policy>;

Learner learner;
learner.setLearningRate(0.1);
learner.setDiscountRate(0.9);

Agent agent(environment, start, goal);
learner.learn(agent, Policy{});
```

Because the sampled next action enters the target, exploration risk becomes part of the learned value. Follow the action carried from one update to the next in [`nu_sarsa.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_sarsa.h).

## Q-learning versus SARSA

![Q-table](assets/q-table.png)

| Question | Q-learning | SARSA |
| --- | --- | --- |
| Target next action | best estimated action | action selected by current policy |
| Policy relationship | off-policy | on-policy |
| Typical behavior near risky exploratory moves | optimistic about eventual greedy behavior | reflects exploration risk |
| nuNN type | `QLearn` | `Sarsa` |

Do not decide from the formula alone. Run both with the same state representation, reward function, policy, start distribution, and episode budget.

## Reward and termination design

In the maze family, a useful transition must define:

- whether the movement is valid;
- the next state after the action;
- reward for ordinary movement, collision, and goal;
- terminal status;
- a maximum episode length to prevent non-terminating exploration.

![Maze rewards](assets/maze-rewards.png)

A positive goal reward with no step cost can find a path without preferring the shortest path. A small step penalty encourages shorter routes but changes the return scale. Reward shaping is part of the model specification and should be documented with results.

## DQN: replacing the table with a network

`Dqn` maps a numeric state vector to one unbounded Q-value per action. Its output layer must therefore use `Activation::Linear`.

Two mechanisms stabilize the neural update:

1. `ExperienceReplayBuffer` stores transitions and returns uniformly sampled mini-batches, reducing temporal correlation.
2. A frozen target network computes bootstrap values and is synchronized from the main network every configured number of learning steps.

The target for the selected action is:

```text
target[action] = reward                                      if done
target[action] = reward + gamma * max Q_target(next_state)   otherwise
```

Other target entries retain the main network's current predictions, so their sample gradients are zero.

## Source-backed DQN loop

This is reduced directly from [`dqn_maze.cc`](https://github.com/eantcal/nunn/blob/main/examples/dqn_maze/dqn_maze.cc):

```cpp
using LC = nu::MlpMatrixNN::LayerConfig;

nu::Dqn agent(
    {LC(2),
     LC(32, nu::Activation::Tanh),
     LC(32, nu::Activation::Tanh),
     LC(4, nu::Activation::Linear)},
    0.005,                      // learning rate
    5000,                       // replay capacity
    32,                         // batch size
    0.99,                       // discount
    100                         // target sync frequency
);

for (int episode = 1; episode <= episodeCount; ++episode) {
    const double epsilon = std::max(
        0.05,
        1.0 - double(episode) / (0.7 * episodeCount)
    );

    int x = startX;
    int y = startY;
    std::vector<double> state = encode(x, y);

    for (int stepCount = 0; stepCount < maxSteps; ++stepCount) {
        const int action = agent.selectAction(state, epsilon);
        auto [nextState, reward, done] = step(x, y, action);

        const double batchLoss =
            agent.learn(state, action, reward, nextState, done);

        if (done)
            break;

        const int nextX = x + deltaX[action];
        const int nextY = y + deltaY[action];
        if (!isWall(nextX, nextY)) {
            x = nextX;
            y = nextY;
        }
        state = nextState;
    }
}
```

`learn()` first stores the transition. It returns `0.0` until the buffer contains one full batch; afterward it returns the pre-update batch MSE.

Read:

- API and invariants in [`nu_dqn.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_dqn.h);
- Bellman target construction and target synchronization in [`nu_dqn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/src/nu_dqn.cc);
- ring-buffer behavior in [`nu_replay_buffer.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_replay_buffer.h);
- tests in [`test_dqn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_dqn.cc).

## Running the examples

| Example | Model | Command | Observe |
| --- | --- | --- | --- |
| [`maze`](https://github.com/eantcal/nunn/blob/main/examples/maze/maze.cc) | Q-learning or SARSA selected in source | `maze` | episode progress and final path |
| [`path_finder`](https://github.com/eantcal/nunn/blob/main/examples/path_finder/path_finder.cc) | tabular graph Q-learning | `path_finder` | learned route through graph states |
| [`dqn_maze`](https://github.com/eantcal/nunn/blob/main/examples/dqn_maze/dqn_maze.cc) | DQN | `dqn_maze [episodes] [learning_rate]` | successes per window, learn-step count, greedy trace |

For DQN, the example normalizes grid coordinates into two numeric inputs and decays epsilon to a floor of 0.05 during the first 70% of episodes. Its final trace sets epsilon to zero.

## Diagnostics that matter

- Report episodic return and success rate, not only Q values or batch loss.
- Cap episode length so failure remains measurable.
- Evaluate with exploration disabled after training.
- Keep state encoding consistent between training and evaluation.
- Verify that terminal transitions do not bootstrap from the next state.
- For DQN, record replay capacity, batch size, epsilon schedule, discount, and target-sync frequency.
- Run several seeds: a single successful trajectory is weak evidence in a stochastic algorithm.

## Keep reading

Use [Training and Diagnostics](Training-and-Diagnostics) for loss and reproducibility checks, or [Implementation Map](Implementation-Map) to move directly among policies, learners, tests, and demos.
