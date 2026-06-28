//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Self-Organizing Map (SOM / Kohonen map).
//
// Architecture:
//   Rectangular grid of rows x cols neurons; each neuron i holds a weight
//   vector w_i in R^d (same dimensionality as the input space).
//
// Online training (Kohonen algorithm):
//   For each input x:
//     1. BMU  = argmin_i  ||x - w_i||
//     2. w_i += lr(t) * h(i, BMU, t) * (x - w_i)   for all i
//   Gaussian neighbourhood:
//     h(i, BMU, t) = exp( -||pos_i - pos_BMU||^2 / (2 * sigma(t)^2) )
//   Exponential decay:
//     lr(t)    = lr_0    * exp(-t / T)
//     sigma(t) = sigma_0 * exp(-t / T)
//
// Internal storage:
//   _W   [rows*cols x inputDim]  — flat weight matrix (row-major neuron index)
//   _pos [rows*cols x 2]         — grid coordinates (row, col) of each neuron
//

#pragma once

#include <Eigen/Core>
#include <random>
#include <utility>
#include <vector>

namespace nu {

class Som {
public:
    // rows, cols   — grid dimensions
    // inputDim     — dimensionality of each input vector
    // lr           — initial learning rate (eta_0); default 0.5
    // initRadius   — initial neighbourhood radius (sigma_0);
    //                0.0 → max(rows, cols) / 2.0
    // seed         — RNG seed for weight initialisation
    // Throws std::invalid_argument if any dimension is zero.
    Som(size_t rows, size_t cols, size_t inputDim, double lr = 0.5, double initRadius = 0.0,
        unsigned seed = 42);

    // Best Matching Unit: grid position (row, col) of the neuron closest to x.
    // Throws std::invalid_argument if x.size() != inputDim().
    std::pair<size_t, size_t> bmu(const std::vector<double>& x) const;

    // Single Kohonen update step with explicit lr and neighbourhood radius.
    void update(const std::vector<double>& x, double lr, double radius);

    // Full online training over dataset for epochs passes.
    //   finalLr     — lr at the last epoch  (exponential schedule lr_0 -> finalLr)
    //   finalRadius — radius at last epoch  (sigma_0 -> finalRadius)
    // The dataset is shuffled each epoch.
    // Throws std::invalid_argument if dataset is empty or input size mismatches.
    void train(const std::vector<std::vector<double>>& dataset, size_t epochs,
        double finalLr = 0.01, double finalRadius = 0.5);

    // Mean Euclidean distance from each sample to its BMU weight vector.
    double quantizationError(const std::vector<std::vector<double>>& dataset) const;

    // Weight vector of neuron at grid position (r, c) as a std::vector<double>.
    // Throws std::out_of_range if r >= rows() or c >= cols().
    std::vector<double> getWeights(size_t r, size_t c) const;

    // Re-randomise all weight vectors uniformly in [0, 1].
    void reshuffleWeights();

    size_t rows() const noexcept { return _rows; }
    size_t cols() const noexcept { return _cols; }
    size_t inputDim() const noexcept { return _dim; }

private:
    size_t _rows, _cols, _dim;
    double _lr0, _r0;
    Eigen::MatrixXd _W; // [rows*cols x inputDim]
    Eigen::MatrixXd _pos; // [rows*cols x 2]
    std::mt19937 _rng;

    size_t _idx(size_t r, size_t c) const noexcept { return r * _cols + c; }
    size_t _bmuFlat(const Eigen::VectorXd& x) const;
};

} // namespace nu
