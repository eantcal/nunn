# Examples Gallery

The examples are part of the documentation. They are small enough to read, but each one isolates a specific learning idea.

## Supervised Basics

| Example | Main model | What it demonstrates |
| --- | --- | --- |
| `and_test` | Perceptron | Linear separation and the perceptron update |
| `xor_test` | `MlpNN` | Why hidden layers are needed for nonlinear separation |
| `mlp_matrix_xor` | `MlpMatrixNN` | The same XOR idea through matrix-based training |
| `linear_regression_demo` | `LinearRegression` | OLS vs gradient descent on linear data |

Use these first when checking a build or reading the library for the first time. They are intentionally tiny: if these do not work, larger demos will only hide the problem.

## MNIST and OCR

| Example | Main model | What it demonstrates |
| --- | --- | --- |
| `mnist_test` | `MlpMatrixNN` / `MlpNN` | Training and evaluating a digit classifier |
| `ocr_test` | `MlpMatrixNN` / `MlpNN` | Interactive drawing, model loading, and MNIST training from GUI |
| `net2json` | conversion tool | Migration from legacy `.net` model files to JSON |

`mnist_test` is the controlled experiment: fixed dataset, known train/test split, reproducible metrics. `ocr_test` is the user-facing experiment: it shows what happens when real drawings differ from the MNIST distribution.

## Recurrent Models

| Example | Main model | What it demonstrates |
| --- | --- | --- |
| `rnn_sine` | RNN / GRU / LSTM | Time-series prediction and hidden state |
| `rnn_char` | RNN / GRU / LSTM | Character-level next-symbol prediction |
| `rnn_adding` | RNN / GRU / LSTM | Memory over a sequence |

These examples are useful for comparing vanilla recurrence with gated recurrence. The important observation is not only final loss, but how quickly each model learns and whether it keeps information over longer spans.

## Convolution and Transformer

| Example | Main model | What it demonstrates |
| --- | --- | --- |
| `cnn_seq` | `ConvNet` | 1D local filters, max pooling, and MLP head |
| `transformer_char` | `MiniTransformer` | Decoder-only self-attention and autoregressive generation |

`cnn_seq` is deliberately 1D so the sliding-window mechanics remain easy to inspect. `transformer_char` avoids external tokenizers and keeps the vocabulary small, making the attention path easier to debug.

## Unsupervised and Representation Learning

| Example | Main model | What it demonstrates |
| --- | --- | --- |
| `hopfield_test` | `HopfieldNN` | Associative recall from noisy or incomplete patterns |
| `kmeans_demo` | `KMeans` | Clustering by nearest centroid |
| `pca_demo` | `Pca` | Linear dimensionality reduction |
| `ae_demo` | `Autoencoder` | Reconstruction through a bottleneck |
| `rbm_demo` | `Rbm` | Probabilistic reconstruction with Contrastive Divergence |
| `vae_demo` | `Vae` | Smooth latent space, reconstruction, and sampling |
| `rbf_demo` | `Rbf` | Distance-based hidden units for function approximation |
| `som_demo` | `Som` | Topological organization of prototypes |

These examples show different meanings of learning without labels: grouping, projection, reconstruction, memory, and self-organization.

## Reinforcement Learning

| Example | Main model | What it demonstrates |
| --- | --- | --- |
| `maze` | Q-learning / SARSA | Tabular value learning in a grid world |
| `path_finder` | Q-learning / SARSA | Path solving on graph-like environments |
| `dqn_maze` | `Dqn` | Neural Q-learning with replay buffer and target network |

The maze examples are best read with the reward function open. In reinforcement learning, the reward design is often as important as the update formula.

## Games and GUI Demos

| Example | Main model | What it demonstrates |
| --- | --- | --- |
| `tictactoe` | `MlpNN` | Console interaction and game-state evaluation |
| `winttt` | `MlpNN` | Windows GUI version of Tic Tac Toe |
| `ocr_test` | `MlpMatrixNN` / `MlpNN` | Windows GUI OCR and model diagnostics |

The GUI demos are useful for packaging and runtime checks because they exercise installed resources, model discovery, icons, launcher behavior, and optional OpenCL runtime availability.

## Suggested Reading Order

1. `and_test`
2. `xor_test`
3. `linear_regression_demo`
4. `mnist_test`
5. `ocr_test`
6. `rnn_sine`
7. `cnn_seq`
8. `transformer_char`
9. `maze`
10. `dqn_maze`

This path moves from simple supervised learning to sequence modeling, structured inputs, attention, and reinforcement learning.
