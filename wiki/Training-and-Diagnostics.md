# Training and Diagnostics

When a model fails, change one assumption at a time. First prove that data shapes and targets are correct; then verify the forward path; only then tune optimization. This page turns that order into a repeatable workflow and points to the nuNN code that defines each metric and update.

![Training loop](assets/training-loop.png)

## The diagnostic loop

1. Establish a simple baseline.
2. Inspect one input, target, and prediction end to end.
3. Overfit a tiny subset.
4. Train on the full training set.
5. Evaluate with updates disabled on held-out data.
6. Save, reload into a fresh object, and compare outputs.
7. Only after correctness is established, compare speed, backend, or optimizer.

A model that cannot overfit a few samples usually has a shape, target, preprocessing, loss, or update-path problem. More epochs rarely repair one of those.

## Verify the data contract first

| Check | Typical failure |
| --- | --- |
| input width equals model input size | wrong flattening, missing feature, stale topology |
| target width equals output size | scalar class used where one-hot vector is required |
| values use the intended scale | bytes left in `[0,255]`, mixed physical units |
| training and inference apply identical preprocessing | OCR drawing resampled differently from MNIST |
| sequence state resets at real boundaries | context leaks across independent samples |
| terminal RL transitions are marked | target incorrectly bootstraps beyond episode end |

Print dimensions and a few values before inspecting gradients.

## Know which loss the code reports

The feedforward loss definitions are in [`nu_costfuncs.h`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_costfuncs.h), [`nu_costfuncs.cc`](https://github.com/eantcal/nunn/blob/main/nunn/common/src/nu_costfuncs.cc), and [`nu_mlpmatrixnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_mlpmatrixnn.cc).

There is one historical scaling difference to remember:

| Method | Current value |
| --- | --- |
| `MlpNN::calcMSE` / common `cf::calcMSE` | `0.5 * sum((output - target)^2)` |
| `MlpMatrixNN::calcMSE` | `sum((output - target)^2) / output_size` |
| both `calcCrossEntropy` implementations | mean binary cross-entropy over outputs |

Do not compare raw MSE numbers or early-stopping thresholds between `MlpNN` and `MlpMatrixNN` without accounting for this scaling. Within one fixed model and output size, the curve remains useful.

For MLP classification, `CrossEntropy` means binary cross-entropy per sigmoid output, not categorical softmax. The constructors reject cross-entropy with a non-sigmoid output layer.

## Read a loss curve

| Shape | Likely interpretation | First checks |
| --- | --- | --- |
| smooth fall, then plateau | normal convergence or capacity limit | held-out metric; then epochs or learning rate |
| oscillation | updates are too aggressive or data order is influential | lower learning rate; inspect scaling; use batches |
| rapid explosion or non-finite value | unstable update, input scale, or target mismatch | learning rate, normalization, gradient clip |
| flat from the first update | no useful gradient or training path not called | target, activation, loss, `backPropagate()` / `trainBatch()` |
| training improves, held-out metric stalls | overfitting or distribution mismatch | split integrity, preprocessing, capacity |
| good before save, bad after load | persistence or file-selection problem | fresh-object round trip and exact filename |
| RL batch loss falls but returns do not improve | Q fit does not imply policy improvement | rewards, termination, exploration, state encoding |

A single final scalar hides whether training was stable. Record at least epoch, training loss, held-out metric, and elapsed time.

## Learning rate

A gradient update has the form:

```text
parameters <- parameters - learning_rate * gradient
```

nuNN implementations may express the stored delta with the opposite sign and add the update; the effective direction is still loss-reducing.

- Too small: the curve falls slowly or appears flat on a short run.
- Too large: the curve oscillates, diverges, or becomes non-finite.
- Changing activation, batch size, or optimizer can change the useful range.

Reduce the learning rate before adding complexity when a previously working topology becomes unstable.

## Momentum and Adam

`MlpNN` and SGD-mode `MlpMatrixNN` can add momentum:

```text
update_t = learning_rate * descent_direction
           + momentum * update_(t-1)
parameter <- parameter + update_t
```

Momentum can accelerate a persistent direction and damp alternating gradients. It can also amplify an excessive learning rate.

Both the Eigen and OpenCL paths in `MlpMatrixNN` additionally support Adam:

```cpp
net.setOptimizer(
    nu::MlpMatrixNN::Optimizer::Adam,
    0.9,
    0.999,
    1e-8
);
```

Calling `setOptimizer()` resets the stored moments and step counter. JSON persistence records whether the model uses SGD or Adam; moment tensors themselves are not part of the current model format.

## Online updates and mini-batches

| Batch size | Effect |
| --- | --- |
| 1 | maximum update noise, simple sample-level inspection |
| small batch | smoother estimate with frequent updates |
| large batch | stable matrix work but fewer parameter updates per epoch |

`MlpMatrixNN::trainBatch(inputs, targets)` averages gradients over the batch. Keep the two containers non-empty and equally sized, and flush a final partial batch.

When comparing batch sizes, decide whether to hold epochs, examples seen, or parameter-update count constant. They are different experimental budgets.

## Early stopping in `NNTrainer`

[`NNTrainer::runTraining`](https://github.com/eantcal/nunn/blob/main/nunn/common/inc/nu_trainer.h) trains complete epochs and compares the mean callback cost for the epoch with `minErr`:

```cpp
nu::MlpTrainer trainer(net, 20000, 0.01);

size_t completedEpochs = trainer.runTraining(
    samples,
    [](nu::MlpNN& model, const auto& target) {
        return model.calcMSE(target);
    }
);
```

A negative `minErr` disables threshold stopping. The optional `p2use` argument clamps to `[0,1]` and selects the leading fraction of the supplied iteration order; it is not a randomized validation split.

The progress callback runs after the current sample update and can stop training by returning `true`. Its behavior is covered by [`test_trainer.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_trainer.cc).

## Activation and target pairing

| Task | Output | Loss |
| --- | --- | --- |
| unbounded regression | linear | MSE |
| bounded scalar reconstruction | depends on data range | MSE or model-specific loss |
| binary / independent labels | sigmoid | binary cross-entropy or MSE |
| one-of-N in MLP | N sigmoid outputs, choose argmax | binary cross-entropy or MSE |
| recurrent classification | softmax `RnnOutput` | cross-entropy |
| DQN action values | linear | Bellman-target MSE |

ReLU and Leaky ReLU are useful hidden activations, but a linear output is necessary when targets can be negative or unbounded. In an MLP, cross-entropy requires sigmoid output by API contract.

## Backend checks

`MlpMatrixNN` exposes the resolved backend through `getBackend()`:

```cpp
auto backend = net.getBackend();
```

`Auto` is a request to choose, not a backend that remains unresolved. After construction it reports Eigen or OpenCL.

For a backend parity check:

1. create two networks with the same topology;
2. copy identical weights and biases with `getLayerW/B()` and `setLayerW/B()`;
3. feed the same input;
4. compare outputs within a floating-point tolerance.

CPU and GPU runs need not be bit-identical. Large disagreement suggests a synchronization, layout, or runtime issue.

## Persistence round trip

Use streams and a fresh object:

```cpp
std::stringstream json;
trained.toJson(json);

nu::MlpNN loaded;
loaded.loadJson(json);

trained.setInputVector(sample);
trained.feedForward();

loaded.setInputVector(sample);
loaded.feedForward();

nu::Vector expected;
nu::Vector actual;
trained.copyOutputVector(expected);
loaded.copyOutputVector(actual);
```

Compare topology, activations, loss choice, and every output within a small tolerance. The repository tests this path for [`MlpNN`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpnn.cc) and [`MlpMatrixNN`](https://github.com/eantcal/nunn/blob/main/tests/test_mlpmatrixnn.cc).

For a file-based check, also log the absolute filename loaded by the application. Many apparent serialization bugs are stale-file bugs.

## Model-specific checks

### Recurrent models

- call `resetState()` before every independent sequence;
- compare teacher-forced one-step metrics with autoregressive rollout;
- record truncation length and gradient clip;
- inspect whether loss is averaged across the whole sequence.

### Convolution

- calculate every channel and length transition before constructing the FC head;
- confirm channel-major flat layout;
- remember that valid convolution shrinks length and pooling discards incomplete remainder windows.

### Transformer

- verify token IDs and fixed context length;
- require `dModel % numHeads == 0`;
- ensure next-token targets are shifted exactly once;
- compare loss with generated samples at a documented temperature.

### Classical and unsupervised models

- standardize before distance- or variance-based methods;
- use the metric defined by the objective;
- inspect centers, components, reconstructions, or map weights, not only one scalar.

### Reinforcement learning

- separate episodic return, success rate, and neural batch loss;
- enforce an episode step limit;
- evaluate greedily;
- run multiple seeds and record the exploration schedule.

## Minimal experiment record

For every result, save:

```text
model and topology
activation and loss
optimizer, learning rate, momentum
batch size / BPTT window / replay settings
training and evaluation sample counts
preprocessing
random seed when exposed
epochs or environment steps
training loss and held-out metric
model filename and source revision
```

That record is usually enough to reproduce a surprising curve or discover the missing difference between two runs.

## Keep reading

Use [Implementation Map](Implementation-Map) to open the exact training source, [MNIST and OCR](MNIST-and-OCR) for a full supervised pipeline, or [Examples Gallery](Examples-Gallery) for controlled experiments.
