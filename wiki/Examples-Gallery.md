# Examples Gallery

Every example is a complete executable built from the repository's [`examples` tree](https://github.com/eantcal/nunn/tree/main/examples). Use this page to choose an experiment, open its source, and know what result to inspect.

## Build and locate examples

Configure once, then build everything or one target:

```sh
cmake -S . -B build -DNUNN_ENABLE_OPENCL=OFF
cmake --build build --config Release
cmake --build build --config Release --target rnn_sine
```

Paths depend on the generator:

```text
single-config:  build/examples/<name>/<name>
Visual Studio:  build\examples\<name>\Release\<name>.exe
```

The source-tree build defines every example below. The current install target contains a smaller runtime subset, so use the build tree when a newer demo is not present in an installed `bin` directory.

## Start with these

| Order | Example | Why it comes next |
| --- | --- | --- |
| 1 | [`and_test`](https://github.com/eantcal/nunn/blob/main/examples/and_test/and_test.cc) | smallest trainable unit and linearly separable data |
| 2 | [`xor_test`](https://github.com/eantcal/nunn/blob/main/examples/xor_test/xor_test.cc) | hidden layer and backpropagation |
| 3 | [`linear_regression_demo`](https://github.com/eantcal/nunn/blob/main/examples/linear_regression_demo/linear_regression_demo.cc) | closed-form fit versus gradient descent |
| 4 | [`kmeans_demo`](https://github.com/eantcal/nunn/blob/main/examples/kmeans_demo/kmeans_demo.cc) | unsupervised assignment and centroids |
| 5 | [`rnn_sine`](https://github.com/eantcal/nunn/blob/main/examples/rnn_sine/rnn_sine.cc) | state, BPTT, and autoregressive evaluation |
| 6 | [`cnn_seq`](https://github.com/eantcal/nunn/blob/main/examples/cnn_seq/cnn_seq.cc) | end-to-end local filters and pooling |
| 7 | [`maze`](https://github.com/eantcal/nunn/blob/main/examples/maze/maze.cc) | state/action/reward loop |
| 8 | [`dqn_maze`](https://github.com/eantcal/nunn/blob/main/examples/dqn_maze/dqn_maze.cc) | replay buffer and target network |
| 9 | [`mnist_test`](https://github.com/eantcal/nunn/blob/main/examples/mnist_test/mnist_test.cc) | full dataset, test metrics, persistence, backend |

## Feedforward and regression

| Target | Model | Run | Look for |
| --- | --- | --- | --- |
| [`and_test`](https://github.com/eantcal/nunn/blob/main/examples/and_test/and_test.cc) | `Perceptron` | `and_test` | all four truth-table rows classified correctly |
| [`xor_test`](https://github.com/eantcal/nunn/blob/main/examples/xor_test/xor_test.cc) | `MlpNN` | `xor_test` | nonlinear separation after training |
| [`counter_test`](https://github.com/eantcal/nunn/blob/main/examples/counter_test/counter_test.cc) | `MlpNN` | `counter_test` | encoded counter-state mapping |
| [`linear_regression_demo`](https://github.com/eantcal/nunn/blob/main/examples/linear_regression_demo/linear_regression_demo.cc) | `LinearRegression` | `linear_regression_demo` | OLS and gradient-descent coefficients, MSE, R² |
| [`titanic`](https://github.com/eantcal/nunn/blob/main/examples/titanic/titanic.cc) | supervised classifier | `titanic` | feature encoding on an embedded real-world table |

AND and XOR form the most useful pair: the data changes only from linearly separable to nonlinearly separable, so the need for a hidden layer is isolated.

## MNIST, OCR, and model tools

| Target | Purpose | Run |
| --- | --- | --- |
| [`mnist_test`](https://github.com/eantcal/nunn/blob/main/examples/mnist_test/mnist_test.cc) | train/test `MlpNN` or `MlpMatrixNN` on IDX data | `mnist_test -p /path/to/mnist` |
| [`ocr_test`](https://github.com/eantcal/nunn/tree/main/examples/ocr_test) | Windows drawing UI, model loading, and MNIST training | launch `ocr_test.exe` |
| [`nunn_topo`](https://github.com/eantcal/nunn/blob/main/nunn_topo/nunn_topo.cc) | render topology from JSON, legacy `.net`, or explicit sizes | `nunn_topo --topology 2,3,1` |
| [`net2json`](https://github.com/eantcal/nunn/blob/main/tools/net2json/net2json.cc) | convert legacy network streams to JSON | `net2json <legacy.net> <model.json>` |

Useful topology commands:

```sh
nunn_topo --topology 2,3,1 --save xor.dot
nunn_topo --load model.json --save model.svg
nunn_topo --load model.net --save model.png
```

DOT needs no external renderer. SVG, PNG, and PDF require Graphviz. Large networks are compacted by default; use `--full` only when every node and edge is genuinely useful.

See [MNIST and OCR](MNIST-and-OCR) for dataset layout and the full option matrix.

## Recurrent models

| Target | Models | Defaults and useful variants |
| --- | --- | --- |
| [`rnn_sine`](https://github.com/eantcal/nunn/blob/main/examples/rnn_sine/rnn_sine.cc) | Vanilla RNN, GRU, LSTM | `rnn_sine`, `rnn_sine --gru`, `rnn_sine --lstm` |
| [`rnn_adding`](https://github.com/eantcal/nunn/blob/main/examples/rnn_adding/rnn_adding.cc) | all three side by side | `rnn_adding [sequence_length] [hidden] [epochs] [lr]` |
| [`rnn_char`](https://github.com/eantcal/nunn/blob/main/examples/rnn_char/rnn_char.cc) | Vanilla RNN, GRU, LSTM | `rnn_char --gru 1200 128 120 0.8` |

`rnn_sine` reports an autoregressive rollout, not only a one-step fit. `rnn_adding` is the better long-memory comparison because exactly two marked values must survive irrelevant steps. `rnn_char` shows how temperature changes sampling from softmax output.

Read [Recurrent Networks](Recurrent-Networks) before interpreting architecture differences.

## Convolution and attention

| Target | Model | Run | Look for |
| --- | --- | --- | --- |
| [`cnn_seq`](https://github.com/eantcal/nunn/blob/main/examples/cnn_seq/cnn_seq.cc) | `ConvNet` | `cnn_seq [epochs] [lr]` | train/test accuracy on noisy one- versus two-cycle signals |
| [`transformer_char`](https://github.com/eantcal/nunn/blob/main/examples/transformer_char/transformer_char.cc) | `MiniTransformer` | `transformer_char [epochs] [lr] [generated_length]` | mean token loss and causal continuation |

These examples keep inputs deliberately small so the full forward and backward implementations in [`nu_conv.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_conv.cc) and [`nu_transformer.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_transformer.cc) remain practical to inspect.

## Classical, representation, and generative models

| Target | Model | Run | Primary metric or artifact |
| --- | --- | --- | --- |
| [`kmeans_demo`](https://github.com/eantcal/nunn/blob/main/examples/kmeans_demo/kmeans_demo.cc) | `KMeans` | `kmeans_demo [k] [samples_per_cluster] [seed]` | inertia, centroids, ASCII assignment map |
| [`pca_demo`](https://github.com/eantcal/nunn/blob/main/examples/pca_demo/pca_demo.cc) | `Pca` | `pca_demo [components] [samples] [seed]` | explained variance and reconstruction MSE |
| [`hopfield_test`](https://github.com/eantcal/nunn/blob/main/examples/hopfield_test/hopfield_test.cc) | `HopfieldNN` | `hopfield_test` | recall from a corrupted binary pattern |
| [`ae_demo`](https://github.com/eantcal/nunn/blob/main/examples/ae_demo/ae_demo.cc) | `Autoencoder` | `ae_demo [epochs]` | bottleneck codes and reconstruction |
| [`rbf_demo`](https://github.com/eantcal/nunn/blob/main/examples/rbf_demo/rbf_demo.cc) | `Rbf` | `rbf_demo [centers] [epochs] [lr]` | sine-regression train/test MSE |
| [`rbm_demo`](https://github.com/eantcal/nunn/blob/main/examples/rbm_demo/rbm_demo.cc) | `Rbm` | `rbm_demo` | reconstruction before/after CD-1 |
| [`vae_demo`](https://github.com/eantcal/nunn/blob/main/examples/vae_demo/vae_demo.cc) | `Vae` | `vae_demo` | reconstructions, latent means, generated samples |
| [`som_demo`](https://github.com/eantcal/nunn/blob/main/examples/som_demo/som_demo.cc) | `Som` | `som_demo` | quantization error and organized prototype grid |

Do not rank these models by one shared loss: each optimizes a different objective. [Classical and Unsupervised Models](Classical-and-Unsupervised) explains the metrics and provides minimal API fragments.

## Reinforcement learning and games

| Target | Model | Run | Observe |
| --- | --- | --- | --- |
| [`maze`](https://github.com/eantcal/nunn/blob/main/examples/maze/maze.cc) | `QLearn` or `Sarsa` selected in source | `maze` | learned navigation policy |
| [`path_finder`](https://github.com/eantcal/nunn/blob/main/examples/path_finder/path_finder.cc) | graph Q-learning | `path_finder` | route recovered from state values |
| [`dqn_maze`](https://github.com/eantcal/nunn/blob/main/examples/dqn_maze/dqn_maze.cc) | `Dqn` | `dqn_maze [episodes] [lr]` | rolling successes, learn steps, greedy trace |
| [`tictactoe`](https://github.com/eantcal/nunn/blob/main/examples/tictactoe/tictactoe.cc) | `MlpNN` | `tictactoe` | console game-state evaluation |
| [`winttt`](https://github.com/eantcal/nunn/tree/main/examples/winttt) | `MlpNN` | launch `winttt.exe` on Windows | GUI inference and packaged resources |

For RL, record the reward definition and episode cap before comparing results. See [Reinforcement Learning](Reinforcement-Learning).

## How to study one example

For any target:

1. Run it unchanged and save the output.
2. Read its `main()` from construction through evaluation.
3. Open the linked public header and identify every method called.
4. Follow one forward/update path in the implementation.
5. Open the matching test from [`tests`](https://github.com/eantcal/nunn/tree/main/tests).
6. Change one parameter, predict the effect, then rerun.

This turns the examples into controlled experiments rather than isolated demos.

## Keep reading

Use [Implementation Map](Implementation-Map) for model-to-source-to-test cross-references and [Training and Diagnostics](Training-and-Diagnostics) for a repeatable experiment record.
