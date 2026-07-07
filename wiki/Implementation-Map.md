# Implementation Map

This page maps the main wiki topics to the nuNN source tree. It is useful when moving from theory to code.

## Neural Networks

| Topic | Main classes | Source |
| --- | --- | --- |
| Perceptron | `Perceptron` | `nunn/neural_networks/inc/nu_perceptron.h` |
| Classic MLP | `MlpNN` | `nunn/neural_networks/inc/nu_mlpnn.h` |
| Matrix MLP | `MlpMatrixNN` | `nunn/neural_networks/inc/nu_mlpmatrixnn.h` |
| Autoencoder | `Autoencoder` | `nunn/neural_networks/inc/nu_autoencoder.h` |
| RBF network | `Rbf` | `nunn/neural_networks/inc/nu_rbf.h` |
| Hopfield | `HopfieldNN` | `nunn/neural_networks/inc/nu_hopfieldnn.h` |
| Linear regression | `LinearRegression` | `nunn/neural_networks/inc/nu_linear_regression.h` |
| K-Means | `KMeans` | `nunn/neural_networks/inc/nu_kmeans.h` |
| PCA | `Pca` | `nunn/neural_networks/inc/nu_pca.h` |
| RBM | `Rbm` | `nunn/neural_networks/inc/nu_rbm.h` |
| VAE | `Vae` | `nunn/neural_networks/inc/nu_vae.h` |
| SOM | `Som` | `nunn/neural_networks/inc/nu_som.h` |

## Sequence Models

| Topic | Main classes | Source |
| --- | --- | --- |
| Vanilla RNN | `VanillaRnn` | `nunn/neural_networks/inc/nu_rnn.h` |
| GRU | `Gru` | `nunn/neural_networks/inc/nu_gru.h` |
| LSTM | `Lstm` | `nunn/neural_networks/inc/nu_lstm.h` |
| Transformer | `MiniTransformer`, `TransformerBlock`, `SelfAttentionLayer`, `LayerNorm` | `nunn/neural_networks/inc/nu_transformer.h` |

## Convolution

| Topic | Main classes | Source |
| --- | --- | --- |
| 1D convolution | `Conv1DLayer` | `nunn/neural_networks/inc/nu_conv.h` |
| 1D max pooling | `MaxPool1DLayer` | `nunn/neural_networks/inc/nu_conv.h` |
| Convolutional pipeline | `ConvNet` | `nunn/neural_networks/inc/nu_convnet.h` |

## MNIST and OCR

| Topic | Main files |
| --- | --- |
| IDX parser and digit conversion | `mnist/mnist.h`, `mnist/mnist.cc` |
| Command-line MNIST training | `examples/mnist_test/mnist_test.cc` |
| Windows OCR GUI | `examples/ocr_test/ocr_test.cpp` |
| OCR launcher/fallback | `examples/ocr_test/ocr_launcher.cpp` |
| Topology visualization | `nunn_topo/nunn_topo.cc` |

## Reinforcement Learning

| Topic | Main classes | Source |
| --- | --- | --- |
| Q-learning | `QLearn` | `nunn/reinforcement/inc/nu_qlearn.h` |
| SARSA | `Sarsa` | `nunn/reinforcement/inc/nu_sarsa.h` |
| Replay buffer | `ExperienceReplayBuffer` | `nunn/reinforcement/inc/nu_replay_buffer.h` |
| DQN | `Dqn` | `nunn/reinforcement/inc/nu_dqn.h` |

## Example Programs

| Area | Examples |
| --- | --- |
| Logic and MLP | `and_test`, `xor_test`, `mlp_matrix_xor` |
| MNIST/OCR | `mnist_test`, `ocr_test` |
| Recurrent models | `rnn_sine`, `rnn_char`, `rnn_adding` |
| Convolution | `cnn_seq` |
| Transformer | `transformer_char` |
| Reinforcement learning | `maze`, `path_finder`, `dqn_maze` |
| Utilities | `net2json`, `nunn_topo`, `nunn_tests` |
