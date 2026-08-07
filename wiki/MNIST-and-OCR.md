# MNIST and OCR

nuNN includes a command-line MNIST experiment and a Windows drawing application. Together they cover the complete path from binary dataset files to normalized vectors, mini-batch training, held-out accuracy, JSON persistence, interactive inference, and topology visualization.

![MNIST MLP OCR](assets/mnist-mlp-ocr.png)

## Data representation

Each MNIST image is 28 by 28 grayscale pixels. `DigitData::toVect()` flattens the row-major bytes and normalizes them to `[0, 1]`.

![MNIST digit](assets/mnist-digit.png)

The complete conversion in [`mnist.cc`](https://github.com/eantcal/nunn/blob/main/mnist/mnist.cc) is:

```cpp
void DigitData::toVect(nu::Vector& values) const noexcept
{
    values.resize(data().size());

    for (size_t i = 0; i < data().size(); ++i)
        values[i] = double((unsigned char)data()[i]) / 255.0;
}
```

Thus:

```text
28 x 28 bytes -> 784 floating-point inputs in [0, 1]
```

![MNIST vectorization](assets/mnist-flatten-vector.png)

`labelToTarget()` produces ten entries with one at the label index:

```cpp
target.resize(10);
std::fill(target.begin(), target.end(), 0.0);
target[getLabel() % 10] = 1.0;
```

A class index is categorical, not a continuous scalar. One-hot targets prevent the model from interpreting digit 8 as numerically close to digit 9.

## Loading IDX files

`TrainingData` receives the label and image filenames, then `load()`:

1. verifies label magic `0x00000801` and image magic `0x00000803`;
2. checks that image and label counts match;
3. reads row and column dimensions;
4. pairs every label with one `DigitData` image.

Public declarations are in [`mnist.h`](https://github.com/eantcal/nunn/blob/main/mnist/mnist.h); parsing is in [`mnist.cc`](https://github.com/eantcal/nunn/blob/main/mnist/mnist.cc).

The dataset directory must contain the four names used by default:

```text
train-labels.idx1-ubyte
train-images.idx3-ubyte
t10k-labels.idx1-ubyte
t10k-images.idx3-ubyte
```

Custom filenames are available through the command-line options shown by `mnist_test --help`.

## `mnist_test` defaults

The current defaults come directly from [`mnist_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/mnist_test/mnist_test.cc):

| Setting | Default |
| --- | --- |
| model | `MlpMatrixNN` |
| topology | `784 -> 300 -> 10` |
| hidden activation | sigmoid |
| output activation | sigmoid |
| cost | MSE |
| learning rate | 0.025 |
| momentum | 0.50 |
| epochs | 100 |
| batch size | 100 |
| backend | `Auto` |

Start a default run with:

```sh
mnist_test -p /path/to/mnist
```

The matrix model is the default because the 60,000-sample training set benefits from batched matrix operations.

## Useful experiment matrix

```sh
# Classic neuron-by-neuron MLP and online updates
mnist_test -p /path/to/mnist --mlp

# Matrix MLP, force Eigen/CPU
mnist_test -p /path/to/mnist --backend cpu

# Matrix MLP, require ArrayFire/OpenCL
mnist_test -p /path/to/mnist --backend opencl

# Matrix MLP with online SGD
mnist_test -p /path/to/mnist --batch 1

# Cross-entropy, ReLU hidden layer, custom topology and schedule
mnist_test -p /path/to/mnist \
  --use_cross_entropy \
  --activation relu \
  --hidden_layer 300 \
  --hidden_layer 100 \
  --learningRate 0.01 \
  --epoch_cnt 50
```

Each `--hidden_layer` occurrence appends a hidden layer. The output remains sigmoid so it is compatible with MSE or binary cross-entropy. The class prediction is the index of the largest of the ten outputs.

## The actual mini-batch path

The matrix training loop converts `DigitData` objects into two parallel batches:

```cpp
std::vector<std::vector<double>> batchInputs;
std::vector<std::vector<double>> batchTargets;

for (const auto& digit : trainingSet.data()) {
    nu::Vector input;
    nu::Vector target;

    digit->toVect(input);
    digit->labelToTarget(target);

    batchInputs.push_back(input.to_stdvec());
    batchTargets.push_back(target.to_stdvec());

    if (batchInputs.size() == batchSize) {
        net.trainBatch(batchInputs, batchTargets);
        batchInputs.clear();
        batchTargets.clear();
    }
}

if (!batchInputs.empty())
    net.trainBatch(batchInputs, batchTargets);
```

The final partial batch must be flushed. Omitting it silently drops samples whenever the dataset size is not an exact multiple of the batch size.

At each epoch the example reshuffles training data, reports epoch time and throughput, evaluates the test set, and keeps the best observed error rate.

## Reading the metrics

The executable reports:

- error and success rate on test digits;
- MSE and cross-entropy calculated from test outputs;
- change in MSE from the previous epoch;
- epoch duration and samples per second;
- best error rate and the epoch that achieved it.

Training updates use the selected cost function. Reporting both losses does not mean both are optimized simultaneously.

The test set must never be passed to `backPropagate()` or `trainBatch()`. Its role is to estimate generalization after training updates.

## Save, load, and inspect

Both MLP paths use JSON stream persistence in the current implementation:

```sh
mnist_test -p /path/to/mnist --save mnist.json
mnist_test -p /path/to/mnist --load mnist.json --skip_training
```

A saved model should contain topology, activation choices, cost function, weights, and biases. Validate a round trip by comparing outputs from the trained object and a fresh loaded object on the same normalized digit.

Visualize the saved model with [`nunn_topo`](https://github.com/eantcal/nunn/blob/main/nunn_topo/nunn_topo.cc):

```sh
nunn_topo --load mnist.json --save mnist.svg
nunn_topo --load mnist.json --save mnist.dot
```

DOT output works without Graphviz. SVG, PNG, and PDF require `dot` in `PATH`. Large topologies use a compact representation by default; `--full` draws every node and edge.

## `ocr_test`

The Windows application in [`examples/ocr_test`](https://github.com/eantcal/nunn/tree/main/examples/ocr_test) can:

- load legacy `.net` and JSON MLP models;
- resample a drawing into a 28 by 28 input;
- show the ten output activations and predicted digit;
- train `MlpNN` or `MlpMatrixNN` from the MNIST files;
- select `Auto`, `CPU`, or `OpenCL` for the matrix model;
- chart cost convergence during training;
- remember dataset and save paths.

The important engineering boundary is preprocessing:

```text
mouse strokes
  -> canvas
  -> resampled 28 x 28 grayscale image
  -> 784 normalized values
  -> same forward pass used by mnist_test
```

A model can score well on MNIST and still struggle with drawings whose centering, scale, stroke width, or contrast differs from the training distribution.

## Model discovery in installed layouts

The GUI searches model locations used by the build and packages, including:

```text
bin/models
bin
../share/nunn/nets
../share/nunn/models
```

On eligible Windows OpenCL builds, `ocr_launcher.cpp` selects an OpenCL-capable executable when the runtime is usable and falls back to the CPU executable otherwise. This keeps missing optional runtime DLLs from becoming a silent startup failure.

Relevant sources:

- training, inference, charting, settings: [`ocr_test.cpp`](https://github.com/eantcal/nunn/blob/main/examples/ocr_test/ocr_test.cpp);
- runtime selection: [`ocr_launcher.cpp`](https://github.com/eantcal/nunn/blob/main/examples/ocr_test/ocr_launcher.cpp);
- packaging rules: [`CMakeLists.txt`](https://github.com/eantcal/nunn/blob/main/CMakeLists.txt).

## Reproducible MNIST/OCR checklist

1. Record model type, hidden layers, activation, loss, learning rate, momentum, batch size, backend, and epoch count.
2. Confirm that every input has 784 values in `[0, 1]` and every target has ten entries.
3. Shuffle only the training data.
4. Evaluate on the untouched test set after each epoch or at a fixed interval.
5. Save the best model to an explicit filename.
6. Load it into a fresh object and compare outputs for the same test digit.
7. Confirm that `ocr_test` reports the intended filename and a 784-input topology.
8. Diagnose drawing-domain mismatch separately from model persistence.

## Keep reading

Use [Training and Diagnostics](Training-and-Diagnostics) for convergence patterns and [Neural Networks](Neural-Networks) for the exact MLP, batch, backend, and persistence APIs.
