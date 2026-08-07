# Theory Notes

This page is the mathematical bridge between the companion book and the nuNN source. It gives the minimum notation needed to recognize each implementation; model-specific pages provide runnable API fragments and diagnostics.

## From equation to file

| Idea | Equation or operation | nuNN code |
| --- | --- | --- |
| activation and derivative | `a = f(z)`, `da/dz` | [`nu_activation.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_activation.h) |
| MSE and binary cross-entropy | `L(output, target)` | [`nu_costfuncs.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_costfuncs.h), [`nu_costfuncs.cc`](https://github.com/eantcal/nunn/blob/main/nunn/common/src/nu_costfuncs.cc) |
| affine layer and backpropagation | `z = W a + b` | [`nu_mlpnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_mlpnn.cc), [`nu_mlpmatrixnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_mlpmatrixnn.cc) |
| recurrent state and BPTT | `h_t = f(x_t, h_(t-1))` | [`nu_rnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_rnn.cc) |
| local shared filters | valid 1D convolution | [`nu_conv.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_conv.cc) |
| scaled dot-product attention | `softmax(QK^T / sqrt(d_k))V` | [`nu_transformer.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_transformer.cc) |
| temporal-difference update | Bellman bootstrap | [`nu_qlearn.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_qlearn.h), [`nu_dqn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/src/nu_dqn.cc) |

## Supervised learning objective

A supervised dataset contains pairs:

```text
(input_i, target_i)
```

A model with parameters `theta` produces `prediction_i = model(input_i; theta)`. Training chooses parameters that reduce an aggregate loss:

```text
J(theta) = average_i L(prediction_i, target_i)
```

Evaluation asks a different question: does the fitted mapping work on samples that did not update `theta`? This is why `mnist_test` separates training and test IDX files.

## Losses in the feedforward code

For one output vector, the common scalar MLP helper currently computes half squared-error sum:

```text
L_half_sum = 0.5 * sum_j (output_j - target_j)^2
```

`MlpMatrixNN::calcMSE()` reports mean squared error over output entries:

```text
L_mean = (1 / output_size) * sum_j (output_j - target_j)^2
```

The scaling difference does not change which parameter values minimize the loss, but it changes numerical curves and thresholds. See [Training and Diagnostics](Training-and-Diagnostics) before comparing the two implementations.

Binary cross-entropy per sigmoid output is:

```text
L_BCE = -(1 / output_size) *
        sum_j [
          target_j * log(output_j)
          + (1 - target_j) * log(1 - output_j)
        ]
```

This is not categorical softmax cross-entropy. The MLPs require sigmoid output with `CostFunction::CrossEntropy`. Recurrent `RnnOutput::Softmax` implements the normalized multiclass case separately.

## Gradient descent and momentum

The gradient points toward fastest increase, so gradient descent moves in the opposite direction:

```text
theta <- theta - learning_rate * gradient(J)
```

Momentum carries a fraction of the previous update:

```text
velocity_t = momentum * velocity_(t-1) - learning_rate * gradient_t
theta <- theta + velocity_t
```

In code, the sign may be absorbed into an error delta and the stored update added to a parameter. To verify the rule, trace one scalar weight and confirm that a small step reduces the loss.

`MlpMatrixNN` implements Adam on both the Eigen and OpenCL paths. Adam keeps exponential moving averages of the gradient and squared gradient, applies bias correction, and divides the first moment by the square root of the second.

## Backpropagation

For layer `l`:

```text
z_l = W_l a_(l-1) + b_l
a_l = f_l(z_l)
```

The chain rule gives a local error signal `delta_l = dJ/dz_l`. For a hidden layer:

```text
delta_l = (W_(l+1)^T delta_(l+1)) elementwise* f_l'(z_l)
```

Then:

```text
dJ/dW_l = delta_l a_(l-1)^T
dJ/db_l = delta_l
```

`MlpNN` stores this at neuron level. `MlpMatrixNN` stores it as matrices and vectors; a batch replaces outer products with matrix products and averages the result.

The local information each layer needs is visible in source: previous activation, current activation or pre-activation, next-layer weights, and incoming error.

## Activation functions

| Activation | Formula | Derivative used by nuNN |
| --- | --- | --- |
| sigmoid | `1 / (1 + exp(-z))` | `a * (1 - a)` |
| tanh | `tanh(z)` | `1 - a^2` |
| ReLU | `max(0, z)` | 1 for positive output, else 0 |
| Leaky ReLU | `z` if positive, else `0.01z` | 1 or 0.01 |
| linear | `z` | 1 |

Sigmoid and tanh can saturate. ReLU avoids positive-side saturation but can stop updating a unit that remains negative. Leaky ReLU retains a small negative slope.

## Generalization and experimental evidence

Optimization and generalization are not the same:

- falling training loss shows that the update fits observed samples;
- improving held-out performance shows transfer to unseen samples;
- a widening gap suggests overfitting or distribution mismatch.

A fair experiment fixes preprocessing and data split, varies one model choice, and reports multiple seeds where randomness matters. [Examples Gallery](Examples-Gallery) suggests controlled comparisons for each family.

## Linear models, clustering, and projection

### Linear regression

```text
prediction = w dot x + b
```

Ordinary least squares solves the full linear system; gradient descent approaches the solution iteratively. The fitted coefficients describe association in the provided features, not causality.

### K-Means

```text
assignment_i = argmin_k ||x_i - centroid_k||^2
centroid_k = mean of samples assigned to k
```

The objective is within-cluster squared distance. It favors roughly compact, similarly scaled groups and requires `k` in advance.

### PCA

After centering `X`:

```text
projected = X_centered V_k
```

The retained right-singular vectors are orthogonal directions of maximum sample variance. High variance is not automatically high predictive value.

## Representation learning

### RBF features

```text
h_j(x) = exp(-||x - center_j||^2 / (2 sigma_j^2))
```

RBF units are local. nuNN fixes centers and widths first, then trains a supervised output layer.

### Autoencoder

```text
x -> encoder -> z -> decoder -> reconstruction
```

A bottleneck encourages compression. Nonlinear activations let the representation go beyond a linear PCA projection.

### RBM

An RBM defines an energy over binary visible and hidden units. Contrastive Divergence approximates the likelihood gradient by contrasting data-driven and short Gibbs-chain statistics.

### VAE

```text
encoder(x) -> mu, log_variance
z = mu + exp(0.5 * log_variance) elementwise* epsilon
```

Reconstruction preserves data while the KL term regularizes the latent distribution toward a standard normal prior. The reparameterization keeps the sampled path differentiable with respect to encoder parameters.

### SOM

A best matching unit and its grid neighbors move toward each sample. The decreasing neighborhood first organizes global structure, then permits local specialization.

## Sequence models

A vanilla recurrent state is:

```text
h_t = tanh(Wx x_t + Wh h_(t-1) + b)
```

Backpropagation through time reuses the same parameters at every unfolded step. The recurrent Jacobian is therefore multiplied repeatedly, causing vanishing or exploding gradients.

GRU and LSTM gates create learned paths that can preserve state. nuNN bounds backward history with truncated BPTT and clamps gradient elements. See [Recurrent Networks](Recurrent-Networks) for the exact shared API.

## Convolution

A convolution applies the same filter at every valid position. Weight sharing reduces parameters and produces translation-equivariant feature maps. `Conv1DLayer` turns windows into columns, multiplies by the filter matrix, then reverses that arrangement during backpropagation.

Max pooling keeps the largest activation in each non-overlapping region and remembers its index so only the winner receives the backward gradient.

## Self-attention

Each token projects to a query, key, and value:

```text
Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
```

The scale limits dot-product growth with dimension. Multiple heads learn different projection spaces. A decoder-only causal mask prevents a position from observing future targets.

Residual paths preserve the input around attention and feed-forward sublayers; Pre-LN normalization makes the normalized signal enter each sublayer.

## Reinforcement learning

Q-learning uses the best estimated next action:

```text
Q(s,a) <- Q(s,a)
          + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
```

SARSA uses the next action actually sampled by the policy:

```text
Q(s,a) <- Q(s,a)
          + alpha * (reward + gamma * Q(s',a') - Q(s,a))
```

DQN replaces the table with a neural approximator and adds experience replay plus a slowly synchronized target network. The target-network prediction is treated as fixed while the main network updates.

Return, exploration, and state distribution make RL evidence noisier than a supervised test loss. Evaluate policy success across episodes and seeds.

## Keep reading

Open [Implementation Map](Implementation-Map) beside this page to trace an equation into code, then use [Training and Diagnostics](Training-and-Diagnostics) to design an experiment that can falsify your expectations.
