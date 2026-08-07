# Implementation Map

Use this page when moving from a concept to the code. Each row links the public contract, implementation, focused test, and smallest relevant example. Header-only templates show “header” in place of a separate source file.

## Core training primitives

| Topic | Public contract / implementation | Tests or use |
| --- | --- | --- |
| numeric vector | [`nu_vector.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_vector.h) · [`nu_vector.cc`](https://github.com/eantcal/nunn/blob/main/nunn/common/src/nu_vector.cc) | [`test_vector.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_vector.cc) |
| activations | [`nu_activation.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_activation.h) | [`test_mlpnn_activations.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpnn_activations.cc) |
| cost functions | [`nu_costfuncs.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_costfuncs.h) · [`nu_costfuncs.cc`](https://github.com/eantcal/nunn/blob/main/nunn/common/src/nu_costfuncs.cc) | [`test_costfuncs.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_costfuncs.cc) |
| generic epoch trainer | [`nu_trainer.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_trainer.h) | [`test_trainer.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_trainer.cc) |
| neuron state | [`nu_neuron.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_neuron.h) | used by `Perceptron` and `MlpNN` |

The smallest complete read is: activation and vector helpers, `Perceptron::feedForward()`, `Perceptron::backPropagate()`, its test, then `and_test`.

## Feedforward networks

| Model | Header | Implementation | Tests | Example |
| --- | --- | --- | --- | --- |
| `Perceptron` | [`nu_perceptron.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_perceptron.h) | [`nu_perceptron.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_perceptron.cc) | [`test_perceptron.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_perceptron.cc) | [`and_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/and_test/and_test.cc) |
| `MlpNN` | [`nu_mlpnn.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_mlpnn.h) | [`nu_mlpnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_mlpnn.cc) | [`test_mlpnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpnn.cc) | [`xor_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/xor_test/xor_test.cc) |
| `MlpMatrixNN` | [`nu_mlpmatrixnn.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_mlpmatrixnn.h) | [`nu_mlpmatrixnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_mlpmatrixnn.cc) | [`test_mlpmatrixnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpmatrixnn.cc) | [`mnist_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/mnist_test/mnist_test.cc) |
| base model abstraction | [`nu_nn_model.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_nn_model.h) | [`nu_nn_model.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_nn_model.cc) | persistence tests above | model loaders |

To compare the two MLP implementations, trace these operations in both source files:

```text
constructor -> weight initialization
setInputVector -> feedForward
output delta -> hidden deltas
weight/bias update
calcMSE / calcCrossEntropy
toJson -> loadJson
```

`MlpMatrixNN` additionally contains batch matrix assembly, backend selection, Adam state, and host/device synchronization.

## Recurrent networks

| Model | Header | Implementation | Tests | Examples |
| --- | --- | --- | --- | --- |
| `VanillaRnn` | [`nu_rnn.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_rnn.h) | [`nu_rnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_rnn.cc) | [`test_rnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rnn.cc) | [`rnn_sine`](https://github.com/eantcal/nunn/blob/main/examples/rnn_sine/rnn_sine.cc), [`rnn_char`](https://github.com/eantcal/nunn/blob/main/examples/rnn_char/rnn_char.cc) |
| `Gru` | [`nu_gru.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_gru.h) | [`nu_gru.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_gru.cc) | [`test_gru.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_gru.cc) | same shared-interface demos |
| `Lstm` | [`nu_lstm.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_lstm.h) | [`nu_lstm.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_lstm.cc) | [`test_lstm.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_lstm.cc) | [`rnn_adding`](https://github.com/eantcal/nunn/blob/main/examples/rnn_adding/rnn_adding.cc) |

Read `step()` before `bptt()`. In the backward path, identify saved per-time-step intermediates, truncation boundaries, gradient clipping, and when state is advanced or reset.

## Convolution and transformer

| Component | Header | Implementation | Tests | Example |
| --- | --- | --- | --- | --- |
| `Conv1DLayer` / `MaxPool1DLayer` | [`nu_conv.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_conv.h) | [`nu_conv.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_conv.cc) | [`test_cnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_cnn.cc) | [`cnn_seq.cc`](https://github.com/eantcal/nunn/blob/main/examples/cnn_seq/cnn_seq.cc) |
| `ConvNet` | [`nu_convnet.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_convnet.h) | [`nu_convnet.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_convnet.cc) | [`test_cnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_cnn.cc) | same |
| `LayerNorm` / `SelfAttentionLayer` / `TransformerBlock` / `MiniTransformer` | [`nu_transformer.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_transformer.h) | [`nu_transformer.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_transformer.cc) | [`test_transformer.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_transformer.cc) | [`transformer_char.cc`](https://github.com/eantcal/nunn/blob/main/examples/transformer_char/transformer_char.cc) |

For convolution, trace channel-major flattening, `im2col`, activation, saved max indices, and reverse gradient routing. For attention, trace shapes through Q/K/V projection, causal masking, row softmax, head concatenation, residuals, and output logits.

## Classical, associative, and representation models

| Model | Header | Implementation | Tests | Example |
| --- | --- | --- | --- | --- |
| `LinearRegression` | [`nu_linear_regression.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_linear_regression.h) | [`nu_linear_regression.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_linear_regression.cc) | [`test_linear_regression.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_linear_regression.cc) | [demo](https://github.com/eantcal/nunn/blob/main/examples/linear_regression_demo/linear_regression_demo.cc) |
| `KMeans` | [`nu_kmeans.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_kmeans.h) | [`nu_kmeans.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_kmeans.cc) | [`test_kmeans.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_kmeans.cc) | [demo](https://github.com/eantcal/nunn/blob/main/examples/kmeans_demo/kmeans_demo.cc) |
| `Pca` | [`nu_pca.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_pca.h) | [`nu_pca.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_pca.cc) | [`test_pca.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_pca.cc) | [demo](https://github.com/eantcal/nunn/blob/main/examples/pca_demo/pca_demo.cc) |
| `HopfieldNN` | [`nu_hopfieldnn.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_hopfieldnn.h) | [`nu_hopfieldnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_hopfieldnn.cc) | MLP/persistence suite and demo assertions | [`hopfield_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/hopfield_test/hopfield_test.cc) |
| `Autoencoder` | [`nu_autoencoder.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_autoencoder.h) | [`nu_autoencoder.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_autoencoder.cc) | [`test_autoencoder.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_autoencoder.cc) | [demo](https://github.com/eantcal/nunn/blob/main/examples/ae_demo/ae_demo.cc) |
| `Rbf` | [`nu_rbf.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_rbf.h) | [`nu_rbf.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_rbf.cc) | [`test_rbf.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rbf.cc) | [demo](https://github.com/eantcal/nunn/blob/main/examples/rbf_demo/rbf_demo.cc) |
| `Rbm` | [`nu_rbm.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_rbm.h) | [`nu_rbm.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_rbm.cc) | [`test_rbm.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rbm.cc) | [demo](https://github.com/eantcal/nunn/blob/main/examples/rbm_demo/rbm_demo.cc) |
| `Vae` | [`nu_vae.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_vae.h) | [`nu_vae.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_vae.cc) | [`test_vae.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_vae.cc) | [demo](https://github.com/eantcal/nunn/blob/main/examples/vae_demo/vae_demo.cc) |
| `Som` | [`nu_som.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_som.h) | [`nu_som.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_som.cc) | [`test_som.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_som.cc) | [demo](https://github.com/eantcal/nunn/blob/main/examples/som_demo/som_demo.cc) |

For these models, start with `fit()` or `train()` and identify exactly which parameters are learned. For example, `Rbf::fitCenters()` fixes centers and widths before `train()` changes only output weights.

## Reinforcement learning

| Component | Header / implementation | Tests | Example |
| --- | --- | --- | --- |
| Q-learning template | [`nu_qlearn.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_qlearn.h) | [`test_rl.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rl.cc) | [`maze.cc`](https://github.com/eantcal/nunn/blob/main/examples/maze/maze.cc) |
| SARSA template | [`nu_sarsa.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_sarsa.h) | [`test_rl.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rl.cc) | [`maze.cc`](https://github.com/eantcal/nunn/blob/main/examples/maze/maze.cc) |
| epsilon-greedy policy | [`nu_e_greedy_policy.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_e_greedy_policy.h) | [`test_rl.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rl.cc) | maze |
| softmax policy | [`nu_softmax_policy.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_softmax_policy.h) | [`test_rl.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rl.cc) | maze |
| graph Q-learning | [`nu_qlgraph.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_qlgraph.h) · [`nu_qlgraph.cc`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/src/nu_qlgraph.cc) | [`test_qmatrix.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_qmatrix.cc) | [`path_finder.cc`](https://github.com/eantcal/nunn/blob/main/examples/path_finder/path_finder.cc) |
| replay buffer | [`nu_replay_buffer.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_replay_buffer.h) | [`test_dqn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_dqn.cc) | DQN maze |
| DQN | [`nu_dqn.h`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/inc/nu_dqn.h) · [`nu_dqn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/reinforcement/src/nu_dqn.cc) | [`test_dqn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_dqn.cc) | [`dqn_maze.cc`](https://github.com/eantcal/nunn/blob/main/examples/dqn_maze/dqn_maze.cc) |

The tabular learners are templates, so their update code lives in the headers. Read the example's `Agent` contract before the learner: state transition and reward semantics belong to the environment, not to `QLearn` or `Sarsa`.

## MNIST, persistence, and tools

| Area | Source | Related executable/test |
| --- | --- | --- |
| IDX parsing and normalized digit vectors | [`mnist.h`](https://github.com/eantcal/nunn/blob/main/mnist/mnist.h) · [`mnist.cc`](https://github.com/eantcal/nunn/blob/main/mnist/mnist.cc) | [`mnist_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/mnist_test/mnist_test.cc) |
| OCR training and recognition | [`ocr_test.cpp`](https://github.com/eantcal/nunn/blob/main/examples/ocr_test/ocr_test.cpp) | `ocr_test` on Windows |
| OCR runtime fallback | [`ocr_launcher.cpp`](https://github.com/eantcal/nunn/blob/main/examples/ocr_test/ocr_launcher.cpp) | packaged launcher |
| legacy-to-JSON conversion | [`net2json.cc`](https://github.com/eantcal/nunn/blob/main/tools/net2json/net2json.cc) | `net2json` |
| topology extraction and Graphviz output | [`nunn_topo.cc`](https://github.com/eantcal/nunn/blob/main/nunn_topo/nunn_topo.cc) | `nunn_topo` |
| persistence behavior | each model source | [`test_mlpnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpnn.cc), [`test_mlpmatrixnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpmatrixnn.cc), [`test_perceptron.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_perceptron.cc) |

## A productive reading order

For any algorithm:

1. Read the public header until you can state input shape, output shape, learned parameters, and error conditions.
2. Open the smallest example and trace construction, training, and evaluation.
3. Follow one forward pass in the source.
4. Follow one update backward, writing down every saved intermediate.
5. Read tests for invalid shapes, numerical behavior, and round trips.
6. Change one example parameter and predict the result before running it.

## Keep reading

Return to [Theory Notes](Theory-Notes) for equations, or [Examples Gallery](Examples-Gallery) for executable-centered study paths.
