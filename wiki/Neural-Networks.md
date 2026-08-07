# Neural Networks

nuNN presents the same feedforward ideas at three levels: a single trainable neuron (`Perceptron`), an explicit neuron-by-neuron multilayer network (`MlpNN`), and a matrix implementation with mini-batches and optional GPU execution (`MlpMatrixNN`). Reading them in that order connects the textbook equations to progressively more practical code.

![MLP topology](assets/mlp-topology.png)

## One model, three views

| Type | Best use | Update style | Source |
| --- | --- | --- | --- |
| `Perceptron` | linear separation and the smallest possible learning rule | online | [header](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_perceptron.h) · [implementation](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_perceptron.cc) · [demo](https://github.com/eantcal/nunn/blob/main/examples/and_test/and_test.cc) |
| `MlpNN` | studying forward propagation and backpropagation at neuron level | online | [header](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_mlpnn.h) · [implementation](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_mlpnn.cc) · [demo](https://github.com/eantcal/nunn/blob/main/examples/xor_test/xor_test.cc) |
| `MlpMatrixNN` | larger datasets, mini-batches, Eigen/OpenCL, Adam | online or mini-batch | [header](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_mlpmatrixnn.h) · [implementation](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_mlpmatrixnn.cc) · [tests](https://github.com/eantcal/nunn/blob/main/tests/test_mlpmatrixnn.cc) |

## The shared computation

For each non-input layer:

```text
z_l = W_l a_(l-1) + b_l
a_l = f_l(z_l)
```

Without `f_l`, stacked affine layers collapse into one affine transformation. The activation is what lets hidden layers form nonlinear features.

The activation dispatch is short enough to read directly in [`nu_activation.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_activation.h):

```cpp
switch (activation) {
case Activation::Sigmoid:   return 1.0 / (1.0 + std::exp(-x));
case Activation::Tanh:      return std::tanh(x);
case Activation::ReLU:      return x > 0.0 ? x : 0.0;
case Activation::LeakyReLU: return x > 0.0 ? x : 0.01 * x;
case Activation::Linear:    return x;
}
```

| Activation | Typical role | Watch for |
| --- | --- | --- |
| `Sigmoid` | bounded binary outputs, small didactic networks | saturation far from zero |
| `Tanh` | zero-centered hidden state or small hidden layers | saturation |
| `ReLU` | deeper hidden layers | inactive units on negative inputs |
| `LeakyReLU` | ReLU-like hidden layers with a negative slope | slope is fixed at 0.01 |
| `Linear` | unbounded regression or Q-values | no nonlinear separation |

## Perceptron: the complete learning step

The perceptron computes one sigmoid neuron and applies a configurable step function for a hard decision. The [implementation](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_perceptron.cc) makes the update visible:

```text
y = sigmoid(w dot x + b)
delta = (target - y) * sigmoid'(y)   for MSE
delta = target - y                   for cross-entropy
w += learning_rate * delta * x + momentum * previous_update
```

A compact, source-backed use is:

```cpp
nu::StepFunction step(0.5, 0.0, 1.0);
nu::Perceptron net(
    2,                         // inputs
    0.2,                       // learning rate
    step,
    nu::CostFunction::MSE,
    0.0                        // momentum
);

net.setInputVector({1.0, 0.0});
net.backPropagate(0.0);        // train
net.setInputVector({1.0, 1.0});
net.feedForward();             // infer
double label = net.getSharpOutput();
```

A perceptron learns AND because the positive and negative examples are linearly separable. It cannot learn XOR; [`xor_test`](https://github.com/eantcal/nunn/blob/main/examples/xor_test/xor_test.cc) demonstrates why the next layer matters.

## `MlpNN`: readable backpropagation

The simplest constructor uses a topology vector:

```cpp
nu::MlpNN net(
    {2, 2, 1},                 // input -> hidden -> output
    0.4,                       // learning rate
    0.9,                       // momentum
    nu::CostFunction::MSE
);
```

For per-layer activations, use `LayerConfig`:

```cpp
using LC = nu::MlpNN::LayerConfig;

nu::MlpNN net(
    {LC(2),
     LC(4, nu::Activation::Tanh),
     LC(1, nu::Activation::Sigmoid)},
    0.1,
    0.5,
    nu::CostFunction::MSE
);
```

The input descriptor is element 0; its activation is ignored. Each later element describes a trainable layer.

The XOR training set used by the demo is only four entries:

```cpp
using TrainingSet = std::map<std::vector<double>, std::vector<double>>;

TrainingSet samples{
    {{0, 0}, {0}},
    {{0, 1}, {1}},
    {{1, 0}, {1}},
    {{1, 1}, {0}}
};

nu::MlpTrainer trainer(net, 40000, -1.0);
trainer.runTraining(samples,
    [](nu::MlpNN& model, const auto& target) {
        return model.calcMSE(target);
    });
```

`backPropagate(target)` performs the forward pass needed for training before propagating deltas and updating weights. For inference, call `setInputVector()`, `feedForward()`, then `copyOutputVector()`.

The relevant code path is:

1. [`MlpNN::setInputVector` and `feedForward`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_mlpnn.cc);
2. output and hidden delta calculation in the same implementation;
3. `_updateNeuronWeights` for momentum and parameter updates;
4. [`NNTrainer::runTraining`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_trainer.h) for the epoch loop and epoch-average early stopping.

## `MlpMatrixNN`: batches and backends

`MlpMatrixNN` uses dense Eigen matrices with shape `[output_size x input_size]`. Its topology must be expressed as `LayerConfig` entries:

```cpp
using LC = nu::MlpMatrixNN::LayerConfig;

nu::MlpMatrixNN net(
    {LC(784),
     LC(300, nu::Activation::ReLU),
     LC(10, nu::Activation::Sigmoid)},
    0.025,
    0.5,
    nu::CostFunction::MSE,
    nu::MlpMatrixNN::ComputeBackend::Auto
);
```

A mini-batch is passed as two parallel containers, not as pairs:

```cpp
std::vector<std::vector<double>> inputs;
std::vector<std::vector<double>> targets;

// Fill equally sized containers, then run one averaged update.
net.trainBatch(inputs, targets);
```

`trainBatch()` validates the batch, runs a matrix forward pass, averages gradients over samples, and performs one update. [`mnist_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/mnist_test/mnist_test.cc) shows the complete batching loop, including the final partial batch.

Available backends:

| Backend | Behavior |
| --- | --- |
| `Auto` | prefer ArrayFire/OpenCL when compiled and usable, otherwise Eigen/CPU |
| `Eigen` | force the CPU path |
| `OpenCL` | require ArrayFire/OpenCL and report failure instead of falling back |

The backend is resolved at construction; inspect it with `getBackend()`.

### SGD, momentum, and Adam

The default optimizer is `MlpMatrixNN::Optimizer::SGD`. Momentum is supplied to the constructor. The Eigen backend can switch to Adam:

```cpp
net.setOptimizer(
    nu::MlpMatrixNN::Optimizer::Adam,
    0.9,      // beta1
    0.999,    // beta2
    1e-8      // epsilon
);
```

Calling `setOptimizer()` resets Adam's moment estimates and step counter. The current OpenCL path always uses SGD; this constraint is declared beside the API in [`nu_mlpmatrixnn.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_mlpmatrixnn.h).

## Loss functions and output semantics

nuNN exposes MSE and binary cross-entropy for the feedforward models. The definitions live in [`nu_costfuncs.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_costfuncs.h) and [`nu_costfuncs.cc`](https://github.com/eantcal/nunn/blob/main/nunn/common/src/nu_costfuncs.cc).

Two details prevent common mistakes:

- `CrossEntropy` requires a sigmoid output layer; both MLP implementations reject incompatible output activations.
- This is per-output binary cross-entropy, not a softmax layer. MNIST uses ten sigmoid outputs against a one-hot target and predicts with the largest output index.

For regression, use a linear output with MSE. For a binary or one-hot classification exercise, sigmoid with cross-entropy is available; MSE remains useful when you want the simplest derivation.

## Save and load JSON

Persistence uses streams. This is the actual `MlpNN` API:

```cpp
#include <fstream>

{
    std::ofstream out("xor.json");
    net.toJson(out);
}

nu::MlpNN restored;
{
    std::ifstream in("xor.json");
    restored.loadJson(in);
}
```

`MlpMatrixNN` also implements `toJson()` and `loadJson()`. Its loader restores topology, activations, loss choice, weights, and biases, and synchronizes backend state. See the round-trip checks in [`test_mlpnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpnn.cc) and [`test_mlpmatrixnn.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpmatrixnn.cc).

To inspect a saved topology:

```sh
nunn_topo --load xor.json --save xor.svg
```

DOT output needs no external renderer; SVG, PNG, and PDF require Graphviz.

## Which MLP should you choose?

- Choose `Perceptron` to study linear separation or a one-neuron update.
- Choose `MlpNN` when transparency is more important than throughput.
- Choose `MlpMatrixNN` for mini-batches, larger datasets, Adam, or optional OpenCL.
- Compare both MLPs on MNIST only after keeping topology, activation, loss, and preprocessing identical.

## Keep reading

Continue with [Training and Diagnostics](Training-and-Diagnostics) for convergence checks, [MNIST and OCR](MNIST-and-OCR) for a complete data pipeline, or [Implementation Map](Implementation-Map) for direct header/source/test links.
