# Theory Notes

This section is the bridge between the book material and the nuNN implementation.

The goal is not to reproduce the book verbatim. Each wiki note should be short, visual, and tied to code in the repository.

## Suggested Pages

The following theory pages are good candidates for expansion:

- Gradient descent and learning rate
- MSE and cross-entropy
- Backpropagation in an MLP
- Mini-batch SGD and matrix multiplication
- Activation functions
- MNIST vectorization
- Vanishing gradients and recurrent networks
- GRU and LSTM gates
- Convolution and pooling
- Self-attention and causal masking
- Q-learning vs SARSA
- DQN, replay buffers, and target networks
- PCA and dimensionality reduction
- K-means and clustering

## Formula Examples

Gradient descent:

```text
theta <- theta - eta * grad J(theta)
```

MSE:

```text
J = (1/N) * sum_i (y_i - t_i)^2
```

Sigmoid:

```text
sigma(x) = 1 / (1 + exp(-x))
```

Q-learning update:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

## Image Sources

The current image assets were copied from the book-generated TikZ PNG files and renamed for wiki readability:

- `mlp-topology.png`
- `training-loop.png`
- `mnist-digit.png`
- `mnist-flatten-vector.png`
- `mnist-mlp-ocr.png`
- `conv-filter.png`
- `maxpool.png`
- `autoencoder.png`
- `rnn-unrolled.png`
- `lstm-cell.png`
- `rl-agent-environment.png`
- `maze-rewards.png`
- `q-table.png`

