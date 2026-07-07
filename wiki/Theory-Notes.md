# Theory Notes

This page collects the main ideas used throughout nuNN and connects them to the implementation. It is intentionally shorter than the book: the goal is to give enough theory to read the code with confidence.

## Loss as Training Signal

Training starts from a numerical question: how wrong is the current model? A loss function turns that question into a scalar value. The optimizer then changes the parameters to reduce that value.

For regression and many didactic neural examples, mean squared error is useful because it directly measures the distance between prediction and target:

```text
MSE = (1/N) * sum_i (y_i - t_i)^2
```

For classification, cross-entropy is often a better fit because it rewards high probability on the correct class and penalizes confident wrong predictions.

In nuNN this choice appears in the MLP training code and in the MNIST dialog, where the user can select MSE or cross-entropy.

## Gradient Descent

The gradient points in the direction where the loss increases fastest. Gradient descent moves in the opposite direction:

```text
theta <- theta - eta * grad J(theta)
```

`eta` is the learning rate. If it is too small, learning is slow. If it is too large, the optimizer can oscillate or diverge. This is one reason the MNIST examples expose learning rate, momentum, epochs, and batch size instead of hiding them.

Momentum adds a memory term to the update. It smooths repeated movement in the same direction and can reduce zig-zagging in narrow valleys of the loss surface.

## Backpropagation

Backpropagation is the chain rule applied layer by layer. A training step has three phases:

1. Forward pass: compute activations from input to output.
2. Error pass: compare output with target and compute output deltas.
3. Backward pass: propagate deltas, compute gradients, update weights.

For a fully connected layer:

```text
z_l = W_l * a_(l-1) + b_l
a_l = f(z_l)
```

The output error is pushed backward through the derivative of the activation function and through the transposed weight matrices. The important practical idea is local: each layer only needs its input activation, output activation, derivative, and the error arriving from the next layer.

## Mini-Batch SGD and Matrices

Online SGD updates the model after each sample. Mini-batch SGD accumulates several samples and updates from their average gradient. This gives a less noisy estimate and maps naturally to matrix multiplication.

That is why nuNN has two MLP implementations:

- `MlpNN` keeps the scalar/neuron-level algorithm visible.
- `MlpMatrixNN` stores weights, activations, and batches in matrix form and can use Eigen or ArrayFire/OpenCL.

The MNIST defaults use `MlpMatrixNN`, backend `Auto`, and batch size `100`.

## Generalization

Reducing training loss is not the whole objective. A model can memorize the training set and still perform poorly on unseen examples. MNIST makes this distinction concrete: training updates the weights, while the test set estimates whether the learned representation transfers to new handwritten digits.

Useful diagnostics:

- training loss decreasing: the optimizer is fitting the examples;
- validation/test accuracy improving: the model is generalizing;
- training loss decreasing while test accuracy stalls: possible overfitting or mismatch between model and data.

## Sequence Models

In sequence data, the current input is not enough. A recurrent model keeps a hidden state:

```text
h_t = f(W_x * x_t + W_h * h_(t-1) + b)
y_t = g(W_y * h_t + c)
```

Backpropagation through time unfolds the recurrent computation over several steps. nuNN uses truncated BPTT so the cost stays bounded.

LSTM and GRU add gates. Gates are learned multipliers that decide what to keep, overwrite, or expose. They help gradients travel across longer spans than a vanilla RNN.

## Convolution and Pooling

Convolution learns small filters and reuses them at every position. This gives three useful properties:

- local connectivity: each filter sees a small window;
- weight sharing: the same filter is applied everywhere;
- translation equivariance: if a pattern shifts, the feature response shifts too.

Max pooling keeps the strongest response in a local region. It reduces the representation and makes it less sensitive to small shifts.

nuNN implements the compact 1D version of these ideas through `Conv1DLayer`, `MaxPool1DLayer`, and `ConvNet`.

## Self-Attention

A Transformer replaces recurrence with direct token-to-token interaction. Each token produces:

- a query: what it is looking for;
- a key: what it offers for matching;
- a value: the information returned if it is attended to.

Scaled dot-product attention is:

```text
Attention(Q,K,V) = softmax((Q * K^T) / sqrt(d_k)) * V
```

The nuNN `MiniTransformer` is decoder-only, so it uses a causal mask: a token can attend to itself and earlier tokens, but not to future tokens.

## Reinforcement Learning

Reinforcement learning is not supervised fitting of fixed labels. An agent acts, observes a reward, and updates its estimates from experience.

Q-learning updates toward the best estimated next action:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
```

SARSA updates toward the next action actually selected by the current policy:

```text
Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
```

The notation difference is small, but the behavior is different: Q-learning is off-policy and more optimistic; SARSA is on-policy and can be more conservative when exploration itself is risky.

DQN keeps the Q-learning target but replaces the table with a neural network. nuNN adds the two stabilizers that make the classic DQN idea usable: a replay buffer and a periodically synchronized target network.

