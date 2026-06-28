# Nunn

| Platform | Status |
|---|---|
| Linux | [![Linux Build](https://travis-ci.org/eantcal/nunn.svg?branch=master)](https://travis-ci.org/eantcal/nunn) |

**Nunn** is a free and open-source machine learning library written in **C++20**, distributed under the **MIT License**.

The library aims to be compact, readable, and practical — a codebase you can actually study while experimenting with neural networks and other machine learning algorithms in modern C++.

---

## Table of contents

1. [Features](#features)
2. [Building and testing](#building-and-testing)
3. [Feedforward networks](#feedforward-networks)
   - [Perceptron](#perceptron-nu_perceptronh)
   - [MlpNN — classic MLP](#mlpnn--classic-mlp-nu_mlpnnh)
   - [MlpMatrixNN — Eigen-backed MLP](#mlpmatrixnn--eigen-backed-mlp-nu_mlpmatrixnnh)
4. [Recurrent networks](#recurrent-networks)
   - [VanillaRnn — Elman RNN](#vanillarnn--elman-rnn-nu_rnnh)
   - [GRU — Gated Recurrent Unit](#gru--gated-recurrent-unit-nu_gruh)
   - [LSTM](#lstm-nu_lstmh)
5. [Associative memory](#associative-memory)
   - [Hopfield network](#hopfield-network)
6. [Reinforcement learning](#reinforcement-learning)
7. [Scripts](#scripts)
8. [Demos and tools](#demos-and-tools)

---

## Features

- **Perceptron** — single neuron, MSE or cross-entropy loss, momentum
- **MlpNN** — classic fully connected MLP with per-layer activations
- **MlpMatrixNN** — Eigen 3.4 backed MLP with mini-batch SGD
- **VanillaRnn** — Elman-style RNN with truncated BPTT
- **GRU** — Gated Recurrent Unit with truncated BPTT
- **LSTM** — Long Short-Term Memory with truncated BPTT
- **Hopfield** — energy-based associative memory
- **Q-learning** and **SARSA** reinforcement learning
- 159 GoogleTest unit tests; all network classes are fully tested
- Cross-platform: Windows, Linux, macOS

---

## Building and testing

Requires CMake 3.14+ and a C++20 compiler. All dependencies (Eigen 3.4, GoogleTest) are fetched automatically via FetchContent.

```sh
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

Pass `-DNUNN_BUILD_TESTS=OFF` to skip the test suite.

---

## Feedforward networks

### Perceptron (`nu_perceptron.h`)

The simplest trainable unit: a single neuron with a weighted sum, a bias, and a nonlinearity. It can only learn **linearly separable** functions (e.g. AND, OR) but serves as the foundation for understanding gradient descent.

**Learning rule** (online SGD):

```
ŷ   = σ(w · x + b)
δ   = (t − ŷ) · σ'(ŷ)      MSE loss
δ   = t − ŷ                 cross-entropy + sigmoid (derivative cancels)
w  += lr · δ · x + momentum · Δw_prev
b  += lr · δ
```

```cpp
#include "nu_perceptron.h"

nu::Perceptron p(
    2,                          // number of inputs
    0.1,                        // learning rate
    0.9,                        // momentum
    nu::CostFunction::MSE
);

p.setInputVector({1.0, 0.0});
p.feedForward();
p.backPropagate({1.0});         // target
```

**Demo:** `and_test` — learns the AND function in a handful of epochs.

---

### MlpNN — classic MLP (`nu_mlpnn.h`)

A fully connected multilayer network trained with online SGD and backpropagation. Topology is expressed as a `vector<size_t>` where the first element is the input size, the last is the output size, and everything in between defines hidden layers.

Supported per-layer activations: `Sigmoid`, `Tanh`, `ReLU`, `Linear`.  
Cost functions: `MSE`, `CrossEntropy` (CE requires Sigmoid on the output layer).

```cpp
#include "nu_mlpnn.h"

nu::MlpNN nn(
    {784, 300, 10},             // input → hidden → output
    0.05,                       // learning rate
    0.9,                        // momentum
    nu::CostFunction::CrossEntropy
);

nn.setInputVector(sample);
nn.feedForward();
nn.backPropagate(target);
```

`MlpNNTrainer` wraps the epoch loop with an early-stopping criterion:

```cpp
nu::MlpNNTrainer trainer(nn, /*max_epochs*/ 30, /*min_err*/ 0.01);
trainer.train<TrainingSet>(dataset, costCallback);
```

Model states can be saved and reloaded:

```cpp
nn.save("model.net");
nn.load("model.net");
```

**Demo:** `xor_test` — the classic non-linearly separable problem.  
**Demo:** `mnist_test` — MNIST digit recognition (784→300→10, ~98% accuracy).

#### XOR walkthrough

XOR cannot be solved by a linear model, but a two-layer MLP with a single hidden unit can learn the required non-linear boundary:

```
x1 | x2 | y
---+----+---
 0 |  0 | 0
 0 |  1 | 1
 1 |  0 | 1
 1 |  1 | 0
```

```cpp
nu::MlpNN nn({2, 2, 1}, 0.4, 0.9);

nu::MlpNNTrainer trainer(nn, 20000, 0.01);
trainer.train<TrainingSet>(
    {{{0,0},{0}}, {{0,1},{1}}, {{1,0},{1}}, {{1,1},{0}}},
    [](nu::MlpNN& net, const auto& target) {
        return net.calcMSE(target);
    }
);
```

---

### MlpMatrixNN — Eigen-backed MLP (`nu_mlpmatrixnn.h`)

A drop-in, higher-performance alternative to `MlpNN`. Weights are stored as `Eigen::MatrixXd` tensors, enabling vectorised GEMV (single sample) and GEMM (mini-batch) operations via Eigen 3.4.

The public interface mirrors `MlpNN`; the key addition is `trainBatch()`:

```cpp
#include "nu_mlpmatrixnn.h"

nu::MlpMatrixNN nn(
    {784, 300, 10},
    0.05, 0.9,
    nu::CostFunction::CrossEntropy
);

// Mini-batch SGD — batch is a vector of (input, target) pairs
nn.trainBatch(batch);
```

`mnist_test` exposes both backends via flags:

```sh
mnist_test                           # classic MlpNN, online SGD
mnist_test --matrix                  # MlpMatrixNN, online SGD
mnist_test --matrix --batch 32       # MlpMatrixNN, mini-batch SGD
```

---

## Recurrent networks

Both recurrent classes (`VanillaRnn`, `Lstm`) share the same public interface and are interchangeable. They use **truncated BPTT** (backpropagation through time) with per-element gradient clipping, and support two output modes:

| `RnnOutput` | Loss | Typical use |
|---|---|---|
| `Linear` | MSE | regression, sequence prediction |
| `Softmax` | Cross-entropy | classification, language modelling |

---

### VanillaRnn — Elman RNN (`nu_rnn.h`)

The simplest recurrent architecture. At each time step the hidden state is updated from the current input and the previous hidden state, then passed through an output projection:

```
h_t = tanh(Wx · x_t  +  Wh · h_{t-1}  +  b_h)
y_t = f_out(Wy · h_t  +  b_y)
```

`Wx` maps the input, `Wh` feeds the hidden state back into itself.  
The hidden state carries information forward in time; training "unrolls" the computation graph for `truncate` steps at a time (truncated BPTT).

**Strengths:** simple, fast, good for short dependencies.  
**Weakness:** gradients vanish quickly over long sequences (the LSTM addresses this).

```cpp
#include "nu_rnn.h"

nu::VanillaRnn rnn(
    /*inputSize*/  1,
    /*hiddenSize*/ 32,
    /*outputSize*/ 1,
    /*lr*/         0.005,
    /*gradClip*/   5.0,
    /*outMode*/    nu::RnnOutput::Linear
);

// Training: feed a full sequence, get mean loss and update weights
rnn.resetState();
double loss = rnn.bptt(inputs, targets, /*truncate*/ 25);

// Inference: one step at a time
rnn.step({0.42});
double y = rnn.getOutput()[0];
```

Weight initialisation: Xavier normal for `Wx`, `Wh`, `Wy`; biases zero.  
Gradient clipping is per-element: each gradient component is clamped to `[-gradClip, +gradClip]`.

---

### GRU — Gated Recurrent Unit (`nu_gru.h`)

Introduced by Cho et al. (2014), the GRU simplifies the LSTM by merging the input and forget gates into a single **update gate** and removing the separate cell state.  The result is a model with ~25% fewer parameters that often matches LSTM performance on short-to-medium sequences.

```
r_t = σ(Wr·x_t + Ur·h_{t-1} + b_r)            reset gate
z_t = σ(Wz·x_t + Uz·h_{t-1} + b_z)            update gate
g_t = tanh(Wh·x_t + Uh·(r_t ⊙ h_{t-1}) + b_h) candidate hidden state
h_t = (1 − z_t) ⊙ h_{t-1}  +  z_t ⊙ g_t      new hidden state
y_t = f_out(Wy·h_t + b_y)
```

**Reset gate** `r_t` controls how much of the previous hidden state leaks into the candidate: when `r_t ≈ 0` the candidate ignores history and can write a fresh value; when `r_t ≈ 1` the candidate behaves like a VanillaRnn step.

**Update gate** `z_t` interpolates between the old hidden state and the candidate: `z_t ≈ 0` keeps the previous state unchanged (long-range memory); `z_t ≈ 1` replaces it entirely (fast adaptation).

**Implementation detail:** input weights for all three gates are stacked as `W [3·nh × ni]`. The recurrent weights for r and z share a single GEMV via `Urz [2·nh × nh]`; the candidate recurrent weight `Uh [nh × nh]` is applied separately to `r_t ⊙ h_{t-1}`.

```cpp
#include "nu_gru.h"

nu::Gru gru(
    /*inputSize*/  2,
    /*hiddenSize*/ 32,
    /*outputSize*/ 1,
    /*lr*/         0.005,
    /*gradClip*/   5.0,
    /*outMode*/    nu::RnnOutput::Linear
);

gru.resetState();
double loss = gru.bptt(inputs, targets, /*truncate*/ 25);

gru.step({0.7, 1.0});
double y = gru.getOutput()[0];
```

The `Gru` API is identical to `VanillaRnn` and `Lstm`.

---

### LSTM (`nu_lstm.h`)

The Long Short-Term Memory adds a **cell state** `c_t` — a separate memory line that flows through time with only element-wise operations (no matrix multiply). Three sigmoid **gates** control what information enters, leaves, and is forgotten from the cell:

```
i_t = σ(Wi·x_t + Ui·h_{t-1} + b_i)      input gate  — what to write
f_t = σ(Wf·x_t + Uf·h_{t-1} + b_f)      forget gate — what to erase  (b_f init = 1)
o_t = σ(Wo·x_t + Uo·h_{t-1} + b_o)      output gate — what to expose
g_t = tanh(Wg·x_t + Ug·h_{t-1} + b_g)  cell candidate

c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t      update cell state
h_t = o_t ⊙ tanh(c_t)                   hidden state exposed to output
y_t = f_out(Wy · h_t + b_y)
```

The forget gate bias is initialised to **1** so the network starts by remembering everything, which helps gradients flow at the beginning of training.

**Implementation detail:** the four gate weight matrices are stacked vertically as `W [4·nh × ni]` and `U [4·nh × nh]`, allowing a single GEMV per step (`pre = W·x + U·h + b`) followed by a split into four blocks. This reduces kernel-launch overhead and keeps the hot path cache-friendly.

```cpp
#include "nu_lstm.h"

nu::Lstm lstm(
    /*inputSize*/  28,          // vocabulary size (one-hot)
    /*hiddenSize*/ 64,
    /*outputSize*/ 28,
    /*lr*/         0.005,
    /*gradClip*/   5.0,
    /*outMode*/    nu::RnnOutput::Softmax
);

lstm.resetState();
double loss = lstm.bptt(inputs, targets, /*truncate*/ 25);

lstm.step(oneHot('a'));
char next = sampleFromSoftmax(lstm.getOutput());
```

The `Lstm` API is identical to `VanillaRnn` — `resetState()`, `step()`, `bptt()`, `reshuffleWeights()`, `getOutput()`, `getHidden()` — so the two classes can be used interchangeably (e.g. via a template or the `--lstm` flag in the demos).

---

### Demo: sine-wave prediction (`rnn_sine`)

Trains a recurrent network to predict the next sample of a sine wave given the current one. After training it runs **autoregressively**: each predicted value is fed back as the next input, testing whether the learned dynamics are self-sustaining.

```sh
rnn_sine                         # VanillaRnn, 1500 epochs, hidden=32, lr=0.005
rnn_sine --lstm                  # LSTM, same defaults
rnn_sine --lstm 2000 64 0.003   # --lstm [epochs] [hidden] [lr]
```

---

### Demo: adding problem benchmark (`rnn_adding`)

The **adding problem** (Hochreiter & Schmidhuber, 1997) is a standard benchmark for comparing recurrent architectures on long-range memory.

Each input sequence has length T. Every element is a pair `(value, marker)` where `value ∈ [0, 1]` and `marker ∈ {0, 1}`. Exactly two positions are marked — one in the first half, one in the second half. The network must output the **running cumulative sum of marked values** at each step, reaching the total sum at the final step.

This requires selective memory: values at unmarked positions must be ignored; values at marked positions must be remembered and accumulated, even when far apart in the sequence.

All three architectures (VanillaRnn, GRU, LSTM) are trained on the same dataset and compared side-by-side:

```sh
rnn_adding                        # seq_len=20, hidden=32, epochs=500, lr=0.005
rnn_adding 30 64 800 0.003        # seq_len hidden epochs lr
```

Example output (300 epochs, seq_len=20, hidden=32):

```
Baseline MAE (predict 0.5) = 0.5447

VanillaRnn   epoch 299   train loss 0.00507   test MAE 0.143
GRU          epoch 299   train loss 0.00088   test MAE 0.054
LSTM         epoch 299   train loss 0.00150   test MAE 0.055
```

GRU and LSTM both significantly outperform VanillaRnn on this task, as expected for a problem that requires retaining specific values over many steps.

---

### Demo: character-level language model (`rnn_char`)

Trains a recurrent network to predict the next character given the current one, accumulating context in the hidden (and cell) state. After training it generates text autoregressively, sampling from the softmax output with a temperature parameter.

The corpus is an embedded English pangram repeated several times (~28-character vocabulary); enough to observe the network learning basic character-level statistics within a few hundred epochs.

```sh
rnn_char                              # VanillaRnn, 800 epochs, hidden=64
rnn_char --lstm                       # LSTM
rnn_char --lstm 1200 128 80 0.6      # --lstm [epochs] [hidden] [gen_len] [temperature]
```

Lower temperature → more repetitive but coherent text.  
Higher temperature → more varied but noisier output.

---

## Associative memory

### Hopfield network

A Hopfield network is an **energy-based** recurrent model that stores patterns as stable attractors of a dynamical system. Given a noisy or incomplete input, it converges to the nearest stored pattern by repeatedly applying an update rule until the energy stops decreasing.

The network has `N` fully connected neurons with symmetric weights (no self-connections). Patterns are stored using the Hebb rule:

```
W_ij = (1/N) Σ_p  ξ_i^p · ξ_j^p        (i ≠ j)
```

Retrieval: start from a corrupted input and update neurons asynchronously until convergence.

Storage capacity is approximately `0.138 · N` patterns before retrieval becomes unreliable.

**Demo:** `hopfield_test` — stores and recalls a set of 100-pixel binary images.

![hopfield test](examples/images/hopfield.jpg)

---

## Reinforcement learning

Nunn includes tabular implementations of two fundamental RL algorithms.

### Q-learning

An **off-policy** temporal-difference method. The agent learns an action-value function `Q(s, a)` — the expected cumulative reward for taking action `a` in state `s` — by bootstrapping from the Bellman equation:

```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a')  −  Q(s, a)]
```

The policy is `ε`-greedy: with probability `ε` the agent explores randomly; otherwise it picks `argmax_a Q(s, a)`. Q-learning converges to the optimal policy regardless of which policy is used to collect data (off-policy).

### SARSA

An **on-policy** TD method. The update uses the action actually taken next (`a'`), not the greedy best:

```
Q(s, a) ← Q(s, a) + α · [r + γ · Q(s', a')  −  Q(s, a)]
```

SARSA is more conservative than Q-learning in stochastic environments because it accounts for the exploration policy during learning.

**Demos:**
- [Maze](https://github.com/eantcal/nunn/blob/master/examples/maze/maze.cc) — navigate from start to goal on a grid
- [Path finder](https://github.com/eantcal/nunn/blob/master/examples/path_finder/path_finder.cc) — find shortest paths under obstacles

---

## Scripts

Ready-made scripts for running and comparing all RNN architectures are provided in `scripts/rnn/`.

### PowerShell (Windows)

```
scripts/rnn/powershell/
  _common.ps1      — shared helpers (exe discovery, Run-Example function)
  run_sine.ps1     — sine-wave prediction: train and compare models
  run_char.ps1     — character-level language model: train and generate text
  run_adding.ps1   — adding problem benchmark (all models, single run)
  run_all.ps1      — run all three examples in sequence
```

**Common options** accepted by most scripts:

| Flag | Description |
|------|-------------|
| `-Quick` | Reduce epoch count for a fast smoke-test |
| `-Model vanilla\|gru\|lstm\|all` | Select one or all architectures (default: `all`) |
| `-Epochs N` | Override epoch count |
| `-Hidden N` | Hidden units (default varies per script) |
| `-Lr F` | Learning rate |

**Examples:**

```powershell
# Quick sanity check — all three models, 400 epochs
.\scripts\rnn\powershell\run_sine.ps1 -Quick

# Full sine comparison with custom params
.\scripts\rnn\powershell\run_sine.ps1 -Epochs 2000 -Hidden 64 -Lr 0.003

# Train GRU char model only
.\scripts\rnn\powershell\run_char.ps1 -Model gru -Epochs 1200 -Hidden 128

# Adding problem benchmark (longer sequences)
.\scripts\rnn\powershell\run_adding.ps1 -SeqLen 30 -Hidden 64 -Epochs 800

# Run everything (quick mode)
.\scripts\rnn\powershell\run_all.ps1 -Quick
```

### Bash (Linux / macOS / Git Bash)

Mirror of the PowerShell scripts with `--` style flags:

```
scripts/rnn/bash/
  _common.sh       — shared helpers
  run_sine.sh      — sine-wave prediction
  run_char.sh      — character-level language model
  run_adding.sh    — adding problem benchmark
  run_all.sh       — run all examples
```

```bash
# Quick sanity check
bash scripts/rnn/bash/run_sine.sh --quick

# Full comparison, custom params
bash scripts/rnn/bash/run_sine.sh --epochs 2000 --hidden 64 --lr 0.003

# Single model
bash scripts/rnn/bash/run_char.sh --model gru --epochs 1200

# Adding benchmark with longer sequences
bash scripts/rnn/bash/run_adding.sh --seq-len 30 --hidden 64 --epochs 800

# Run everything
bash scripts/rnn/bash/run_all.sh --quick
```

### MNIST training scripts

Training scripts for the MNIST examples are in `scripts/mnist/powershell/` and `scripts/mnist/bash/`.  
`run_all` trains all activation/cost-function combinations and prints a comparison table with BER and throughput.

```powershell
.\scripts\mnist\powershell\run_all.ps1 -Quick
```

```bash
bash scripts/mnist/bash/run_all.sh --quick
```

---

## Demos and tools

| Demo | Model | Description |
|------|-------|-------------|
| `and_test` | Perceptron | AND function (linearly separable) |
| `xor_test` | MlpNN | XOR function (non-linearly separable) |
| `mnist_test` | MlpNN / MlpMatrixNN | MNIST digit recognition (784→300→10) |
| `ocr_test` | MlpNN | Interactive handwritten digit recognition |
| `rnn_sine` | VanillaRnn / GRU / LSTM | Sine-wave next-step prediction |
| `rnn_char` | VanillaRnn / GRU / LSTM | Character-level language model |
| `rnn_adding` | VanillaRnn / GRU / LSTM | Adding problem benchmark (selective memory) |
| `tictactoe` | MlpNN | Tic Tac Toe via neural network |
| `winttt` | MlpNN | Interactive Windows Tic Tac Toe |
| `hopfield_test` | Hopfield | Pattern recall from noisy input |
| `maze` | Q-learning / SARSA | Grid-world navigation |
| `path_finder` | Q-learning / SARSA | Shortest-path under obstacles |
| `nunn_topo` | — | Export network topology to Graphviz DOT |

### MNIST

The MNIST dataset contains 60,000 training and 10,000 test images of handwritten digits (28×28 grayscale, flattened to 784 inputs). The `mnist_test` tool supports three backends:

```sh
mnist_test -p /path/to/mnist              # MlpNN, online SGD
mnist_test -p /path/to/mnist --matrix     # MlpMatrixNN, online SGD
mnist_test -p /path/to/mnist --matrix --batch 32   # MlpMatrixNN, mini-batch SGD
```

More information: http://yann.lecun.com/exdb/mnist/

### OCR demo

`ocr_test` loads a `.net` model produced by `mnist_test` and performs real-time handwritten digit recognition.

[![Watch the video](https://youtu.be/ereeEG_1lmY)](https://youtu.be/ereeEG_1lmY)

![ocr_test](examples/images/ocr.jpg)

### Topology visualiser (`nunn_topo`)

Exports a network topology to **Graphviz DOT** format, which `dot` can render as GIF, PNG, SVG, or PostScript.
