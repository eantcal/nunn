# nuNN: machine learning you can read

nuNN is a compact C++20 library for learning machine-learning algorithms by following them from equation to implementation. Forward passes, gradients, training loops, persistence, and complete demo programs remain visible in ordinary C++ rather than disappearing behind a framework.

This wiki is the guided layer between the companion book, *Fundamentals of Machine Learning: Algorithms and Applications in C++*, and the [nuNN source tree](https://github.com/eantcal/nunn). Each topic starts from the model, shows the public API in use, and points to the implementation, tests, and runnable examples.

## Choose a path

| If you want to... | Start here | Then run |
| --- | --- | --- |
| Build the project and verify the toolchain | [Getting Started](Getting-Started) | `and_test`, `xor_test`, `nunn_tests` |
| Follow backpropagation from neurons to matrices | [Neural Networks](Neural-Networks) | `xor_test`, `mnist_test` |
| Work with time series or text | [Recurrent Networks](Recurrent-Networks) | `rnn_sine`, `rnn_adding`, `rnn_char` |
| Study local filters or attention | [Convolutional Networks and Transformer](Convolutional-and-Transformer) | `cnn_seq`, `transformer_char` |
| Compare regression, clustering, projection, and generative models | [Classical and Unsupervised Models](Classical-and-Unsupervised) | `linear_regression_demo`, `kmeans_demo`, `pca_demo`, `vae_demo` |
| Learn from rewards instead of labels | [Reinforcement Learning](Reinforcement-Learning) | `maze`, `path_finder`, `dqn_maze` |
| Train and deploy handwritten-digit recognition | [MNIST and OCR](MNIST-and-OCR) | `mnist_test`, `ocr_test`, `nunn_topo` |
| Diagnose a model that does not converge | [Training and Diagnostics](Training-and-Diagnostics) | compare loss, accuracy, and saved-model output |

## Library map

| Family | Main types | Implementation | Smallest useful example |
| --- | --- | --- | --- |
| Feedforward | `Perceptron`, `MlpNN`, `MlpMatrixNN` | [`neural_networks`](https://github.com/eantcal/nunn/tree/main/nunn/neural_networks) | [`and_test`](https://github.com/eantcal/nunn/blob/main/examples/and_test/and_test.cc), [`xor_test`](https://github.com/eantcal/nunn/blob/main/examples/xor_test/xor_test.cc) |
| Sequence | `VanillaRnn`, `Gru`, `Lstm` | [`nu_rnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_rnn.cc), [`nu_gru.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_gru.cc), [`nu_lstm.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_lstm.cc) | [`rnn_sine`](https://github.com/eantcal/nunn/blob/main/examples/rnn_sine/rnn_sine.cc) |
| Convolution and attention | `Conv1DLayer`, `ConvNet`, `MiniTransformer` | [`nu_conv.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_conv.cc), [`nu_transformer.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_transformer.cc) | [`cnn_seq`](https://github.com/eantcal/nunn/blob/main/examples/cnn_seq/cnn_seq.cc), [`transformer_char`](https://github.com/eantcal/nunn/blob/main/examples/transformer_char/transformer_char.cc) |
| Classical and unsupervised | `LinearRegression`, `KMeans`, `Pca`, `Som`, `Rbf`, `Rbm`, `Autoencoder`, `Vae` | [`neural_networks/src`](https://github.com/eantcal/nunn/tree/main/nunn/neural_networks/src) | [examples directory](https://github.com/eantcal/nunn/tree/main/examples) |
| Reinforcement learning | `QLearn`, `Sarsa`, `Dqn` | [`reinforcement`](https://github.com/eantcal/nunn/tree/main/nunn/reinforcement) | [`maze`](https://github.com/eantcal/nunn/blob/main/examples/maze/maze.cc), [`dqn_maze`](https://github.com/eantcal/nunn/blob/main/examples/dqn_maze/dqn_maze.cc) |
| Data and tools | `TrainingData`, `DigitData`, `nunn_topo`, `net2json` | [`mnist`](https://github.com/eantcal/nunn/tree/main/mnist), [`nunn_topo`](https://github.com/eantcal/nunn/tree/main/nunn_topo), [`tools`](https://github.com/eantcal/nunn/tree/main/tools) | `mnist_test`, `nunn_topo` |

The complete cross-reference, including tests for each model, is in the [Implementation Map](Implementation-Map). The [Examples Gallery](Examples-Gallery) groups every executable by learning objective and gives build-tree run commands.

## What the library deliberately exposes

- `MlpNN` keeps neurons, weights, deltas, and online updates explicit; `MlpMatrixNN` expresses the same computation with Eigen matrices and mini-batches.
- Recurrent models expose `resetState()`, `step()`, and truncated `bptt()` so stateful inference and sequence training stay distinct.
- `ConvNet` sends the fully connected head's input gradient back through pooling and convolution.
- `MiniTransformer` contains embeddings, sinusoidal positions, causal multi-head attention, Pre-LN blocks, residual connections, and autoregressive generation.
- `Dqn` shows the replay-buffer and target-network mechanisms that stabilize neural Q-learning.
- JSON and legacy-stream persistence make the full train-save-load-infer path inspectable.

This is an educational and experimental library, not an attempt to replace a large production framework. The compact scope is a feature: the important algorithms fit in files that can be read, modified, and tested end to end.

## Quick start

```sh
git clone https://github.com/eantcal/nunn.git
cd nunn
cmake -S . -B build -DNUNN_ENABLE_OPENCL=OFF
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Then run `build/examples/xor_test/xor_test` on a single-config generator, or `build\examples\xor_test\Release\xor_test.exe` with Visual Studio. See [Getting Started](Getting-Started) for optional OpenCL support, source-backed starter code, and platform-specific paths.

## Companion book

The wiki distills selected material from *Fundamentals of Machine Learning: Algorithms and Applications in C++* and connects it directly to nuNN. It is a practical reference, not a replacement for the book's full derivations and discussion.

- [English Kindle](https://www.amazon.com/dp/B0GY9L7N22)
- [English paperback](https://www.amazon.com/dp/B0H7KQCFJY)
- [Italian Kindle](https://www.amazon.it/dp/B0H6Q12LVJ)
- [Italian paperback](https://www.amazon.it/dp/B0DF69MPZF)

The diagrams under `wiki/assets/` are adapted from the book-generated figures. Source snippets in this wiki are intentionally short; the linked files remain authoritative.

## Keep reading

Start with [Getting Started](Getting-Started), use [Theory Notes](Theory-Notes) when you need the mathematical bridge, and keep [Training and Diagnostics](Training-and-Diagnostics) beside you while experimenting.
