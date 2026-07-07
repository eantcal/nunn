# Neural Networks

nuNN contains two MLP implementations with different educational goals.

`MlpNN` is the classic readable implementation. Neurons, weights, deltas, and updates are explicit, making it useful for understanding backpropagation step by step.

`MlpMatrixNN` is the matrix-oriented implementation. It uses Eigen on CPU and can use ArrayFire/OpenCL when available. It is the preferred implementation for MNIST and mini-batch training.

![MLP topology](assets/mlp-topology.png)

## Perceptron

The perceptron is the smallest supervised neural model in the library. It is useful for linearly separable problems and as a first bridge between linear classification and neural learning.

Implementation:

- `nunn/neural_networks/inc/nu_perceptron.h`
- `nunn/neural_networks/src/nu_perceptron.cc`

Demo:

- `and_test`

## MlpNN

`MlpNN` implements a fully connected multilayer perceptron with online SGD. It supports configurable hidden layers, activations, MSE or cross-entropy, momentum, model save/load, and topology export.

Good uses:

- studying forward propagation;
- studying backpropagation;
- small logic problems such as XOR;
- comparing scalar-style code against the matrix implementation.

## MlpMatrixNN

`MlpMatrixNN` stores layer weights and activations as matrices/vectors. For a layer:

```text
z = W a + b
a_next = f(z)
```

For mini-batches, the same idea becomes a matrix multiplication over many samples. This is why `MlpMatrixNN` is a better fit for acceleration.

![Training loop](assets/training-loop.png)

Current defaults for MNIST training use:

- `MlpMatrixNN`
- backend `Auto`
- batch size `100`

The `Auto` backend uses OpenCL when ArrayFire/OpenCL is available and falls back to Eigen/CPU otherwise.

## RBF Network

`Rbf` implements a radial basis function network. Hidden units are Gaussian responses around centers:

```text
h_j(x) = exp(-||x - c_j||^2 / (2 sigma_j^2))
```

The centers are selected from data, while output weights are trained with SGD.

Demo:

- `rbf_demo`

## Autoencoder

The autoencoder is an encoder-decoder model built on top of `MlpMatrixNN`. It compresses input into a latent representation and reconstructs it.

![Autoencoder](assets/autoencoder.png)

Demo:

- `ae_demo`

