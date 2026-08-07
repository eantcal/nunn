# Classical and Unsupervised Models

This part of nuNN covers models that answer different questions from a classifier: fit a linear relationship, discover groups, retain variance, recall a pattern, compress an input, estimate a probabilistic representation, or organize prototypes on a map. The public APIs stay small enough to compare directly.

## Choose by objective

| Objective | Model | Fit result | Example |
| --- | --- | --- | --- |
| predict a continuous value with a line or plane | `LinearRegression` | coefficients and intercept | [`linear_regression_demo`](https://github.com/eantcal/nunn/blob/main/examples/linear_regression_demo/linear_regression_demo.cc) |
| divide samples into compact groups | `KMeans` | centroids and labels | [`kmeans_demo`](https://github.com/eantcal/nunn/blob/main/examples/kmeans_demo/kmeans_demo.cc) |
| retain high-variance linear directions | `Pca` | components and projections | [`pca_demo`](https://github.com/eantcal/nunn/blob/main/examples/pca_demo/pca_demo.cc) |
| recover a stored binary pattern | `HopfieldNN` | attractor dynamics | [`hopfield_test`](https://github.com/eantcal/nunn/blob/main/examples/hopfield_test/hopfield_test.cc) |
| learn nonlinear reconstruction | `Autoencoder` | latent code and reconstruction | [`ae_demo`](https://github.com/eantcal/nunn/blob/main/examples/ae_demo/ae_demo.cc) |
| fit with distance-to-center features | `Rbf` | supervised output weights | [`rbf_demo`](https://github.com/eantcal/nunn/blob/main/examples/rbf_demo/rbf_demo.cc) |
| model binary data probabilistically | `Rbm` | hidden probabilities and reconstructions | [`rbm_demo`](https://github.com/eantcal/nunn/blob/main/examples/rbm_demo/rbm_demo.cc) |
| learn a smooth generative latent space | `Vae` | latent distribution, reconstruction, samples | [`vae_demo`](https://github.com/eantcal/nunn/blob/main/examples/vae_demo/vae_demo.cc) |
| organize prototypes on a topology-preserving grid | `Som` | best matching units and neuron weights | [`som_demo`](https://github.com/eantcal/nunn/blob/main/examples/som_demo/som_demo.cc) |

## Linear regression

The model is:

```text
prediction = w dot x + b
```

nuNN supports ordinary least squares and gradient descent. OLS uses an Eigen QR solve and is normally the practical choice; gradient descent exposes the iterative update used later by neural models.

Source-backed API:

```cpp
#include "nu_linear_regression.h"

nu::LinearRegression model(nu::LinearRegression::Method::OLS);
model.fit(features, targets);

double prediction = model.predict({1.5, 2.0});
double testMse = model.mse(testFeatures, testTargets);
double testR2 = model.rSquared(testFeatures, testTargets);

const auto& weights = model.coefficients();
double bias = model.intercept();
```

For the iterative path:

```cpp
nu::LinearRegression model(
    nu::LinearRegression::Method::GradientDescent,
    0.05,                       // learning rate
    200000,                     // maximum iterations
    1e-10                       // parameter-change tolerance
);
```

Read the validation and API in [`nu_linear_regression.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_linear_regression.h), the QR and gradient-descent paths in [`nu_linear_regression.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_linear_regression.cc), and edge cases in [`test_linear_regression.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_linear_regression.cc).

Interpret `R²` together with test MSE. A high training `R²` does not establish that the relationship generalizes.

## K-Means

K-Means alternates between nearest-centroid assignment and centroid recomputation. nuNN uses k-means++ initialization and stops when centroid movement is below the tolerance or the iteration limit is reached.

```cpp
#include "nu_kmeans.h"

nu::KMeans model(
    4,                          // clusters
    300,                        // maximum iterations
    1e-6,                       // centroid-shift tolerance
    42                          // initialization seed
);

model.fit(samples);
auto labels = model.predict(samples);
double withinClusterSse = model.inertia(samples);
const auto& centers = model.centroids();
```

The source handles an empty cluster by retaining its previous centroid. Follow initialization, assignment, and update in [`nu_kmeans.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_kmeans.cc); see [`test_kmeans.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_kmeans.cc) for convergence and validation cases.

Because Euclidean distance drives both assignment and inertia, standardize features when their units differ materially. Compare several `k` values and seeds instead of treating one inertia value as self-explanatory.

## PCA

PCA centers the dataset and finds orthogonal directions of decreasing variance. nuNN uses a thin SVD and stores the requested right-singular vectors.

```cpp
#include "nu_pca.h"

nu::Pca pca(2);
pca.fit(samples);

auto projected = pca.transform(samples);
auto reconstructed = pca.inverseTransform(projected.front());
const auto& ratios = pca.explainedVarianceRatio();
double retained = pca.totalExplainedVariance();
```

`nComponents` must not exceed `min(number_of_samples, number_of_features)`. The implementation is in [`nu_pca.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_pca.cc), with round-trip and variance checks in [`test_pca.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_pca.cc).

PCA is linear. If a low-dimensional structure is curved, compare reconstruction with an autoencoder rather than assuming that adding one more principal component solves the representation problem.

## Hopfield associative memory

A classical Hopfield network stores bipolar patterns as attractors using symmetric weights and no self-connections:

```text
W_ij += pattern_i * pattern_j,  i != j
W_ii = 0
```

A compact use is:

```cpp
nu::HopfieldNN memory(8);

memory.addPattern({1, 1, 1, 1, -1, -1, -1, -1});

nu::Vector recalled;
memory.recall(
    {1, 1, -1, 1, -1, -1, -1, -1},
    recalled
);
```

Read [`nu_hopfieldnn.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_hopfieldnn.cc) beside [`hopfield_test.cc`](https://github.com/eantcal/nunn/blob/main/examples/hopfield_test/hopfield_test.cc). The often-cited random-pattern capacity near `0.138 * N` is a rule of thumb, not a guarantee for correlated patterns.

## Autoencoder

`Autoencoder` builds one symmetric `MlpMatrixNN` from an encoder specification. The final encoder size is the bottleneck; the decoder mirrors the earlier sizes and ends with a linear reconstruction layer.

![Autoencoder](assets/autoencoder.png)

The constructor in [`nu_autoencoder.h`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/inc/nu_autoencoder.h) is:

```cpp
nu::Autoencoder model(
    16,                         // input and reconstruction size
    {8, 4},                     // encoder; 4 is the bottleneck
    nu::Activation::Tanh,
    0.005
);

double finalMse = model.train(dataset, 500);
auto latent = model.encode(dataset.front());
auto reconstruction = model.decode(latent);
```

`train()` consumes the full dataset for the requested number of epochs and returns the final epoch's mean reconstruction MSE. `decode()` uses decoder weights synchronized after training or reshuffling.

Follow topology construction and decoder synchronization in [`nu_autoencoder.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_autoencoder.cc); see [`test_autoencoder.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_autoencoder.cc).

## RBF network

An RBF hidden unit responds to distance from a center:

```text
h_j(x) = exp(-||x - center_j||^2 / (2 sigma_j^2))
```

nuNN first samples centers from the data and derives widths, then trains only the output weights:

```cpp
nu::Rbf model(
    1,                          // input dimensions
    12,                         // centers
    1,                          // outputs
    0.05,
    nu::RnnOutput::Linear
);

model.fitCenters(trainInputs);
double finalLoss = model.train(trainInputs, trainTargets, 3000);
auto prediction = model.forward({0.5});
```

Call `fitCenters()` before `forward()` or `train()`. For classification, select `RnnOutput::Softmax` and provide one-hot targets. The full algorithm is in [`nu_rbf.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_rbf.cc) and tested by [`test_rbf.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rbf.cc).

## Restricted Boltzmann machine

An RBM is a bipartite energy-based model: visible units connect to hidden units, but there are no within-layer connections. nuNN trains binary/normalized data with online Contrastive Divergence.

```cpp
nu::Rbm model(
    8,                          // visible units
    6,                          // hidden units
    0.05,
    42                          // RNG seed
);

model.train(dataset, 300, 1);  // CD-1
auto probabilities = model.reconstruct(noisySample);
auto hiddenCode = model.encode(noisySample);
double error = model.reconstructionError(dataset);
```

`reconstruct()` is a soft probability path, while Gibbs sampling is used inside training. The phases and parameter update are in [`nu_rbm.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_rbm.cc); [`test_rbm.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_rbm.cc) covers probabilities, sampling shapes, and learning.

## Variational autoencoder

The VAE predicts `mu` and `log_variance`, samples with the reparameterization trick, and minimizes reconstruction loss plus a KL term:

```text
z = mu + exp(0.5 * log_variance) elementwise* epsilon
epsilon ~ Normal(0, I)
```

Source-backed use from [`vae_demo.cc`](https://github.com/eantcal/nunn/blob/main/examples/vae_demo/vae_demo.cc):

```cpp
nu::Vae model(
    8,                          // input
    32,                         // encoder/decoder hidden width
    4,                          // latent dimensions
    0.003,
    42
);

model.train(dataset, 2000, 0.2); // KL warm-up over first 20%
auto [mu, logVariance] = model.encode(sample);
auto deterministic = model.reconstruct(sample); // decodes mu
auto generated = model.generate();              // z ~ Normal(0, I)
```

The output uses sigmoid and binary cross-entropy, so inputs should be in `[0, 1]`. Read the full gradient flow in [`nu_vae.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_vae.cc) and the loss/shape checks in [`test_vae.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_vae.cc).

KL warm-up prevents the regularizer from overwhelming reconstruction at the beginning of training. Evaluate reconstruction and sample quality separately.

## Self-Organizing Map

A SOM assigns each input to its closest neuron, then moves that best matching unit and its grid neighbors toward the sample:

```text
BMU = argmin_i ||x - w_i||
w_i += learning_rate(t) * neighborhood(i, BMU, t) * (x - w_i)
```

Source-backed use:

```cpp
nu::Som map(
    6, 6,                       // grid
    2,                          // input dimensions
    0.5,                        // initial learning rate
    0.0,                        // auto radius = max(rows, cols) / 2
    42
);

map.train(dataset, 200, 0.01, 0.5);
auto [row, column] = map.bmu(sample);
double quantization = map.quantizationError(dataset);
auto prototype = map.getWeights(row, column);
```

The exponentially decaying learning rate and neighborhood are implemented in [`nu_som.cc`](https://github.com/eantcal/nunn/blob/main/nunn/neural_networks/src/nu_som.cc); [`test_som.cc`](https://github.com/eantcal/nunn/blob/main/tests/test_som.cc) checks BMUs, training, and error reduction.

Unlike K-Means, a SOM preserves a grid relation among prototypes. Quantization error measures closeness to a prototype, not whether the grid's neighborhood relations are meaningful; inspect both.

## A fair comparison checklist

- Scale features before distance-, variance-, or gradient-based fitting.
- Keep training and evaluation data separate even for unsupervised reconstruction metrics.
- Report seeds for K-Means, RBM, VAE, SOM, and any generated dataset.
- Compare the metric that matches the objective: MSE/R², inertia, retained variance, reconstruction error, or quantization error.
- Inspect learned objects—coefficients, centers, components, reconstructions, generated samples, or map weights—not only one scalar.

## Keep reading

Use [Examples Gallery](Examples-Gallery) for commands and expected observations, [Theory Notes](Theory-Notes) for the shared mathematics, and [Training and Diagnostics](Training-and-Diagnostics) when an iterative model is unstable.
