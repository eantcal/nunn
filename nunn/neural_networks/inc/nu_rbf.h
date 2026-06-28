//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// RBF Network (Radial Basis Function Network).
//
// Architecture:
//   Hidden: h_j = exp(-||x - c_j||^2 / (2 * sigma_j^2))   j = 0..nc-1
//   Output: y   = f_out(W_out * h + b_out)
//
// Training procedure:
//   1. Call fitCenters(dataset): fixes center positions c_j (random-subset
//      sampling from data) and widths sigma_j = d_max / sqrt(2 * nc),
//      where d_max is the maximum pairwise distance between centers.
//   2. Call train(): updates only W_out / b_out via online SGD;
//      centers and widths are never modified after fitCenters().
//
// Output mode (RnnOutput, from nu_rnn.h):
//   Linear  — identity activation, MSE loss
//   Softmax — softmax activation, cross-entropy loss

#pragma once

#include "nu_rnn.h"

#include <Eigen/Core>
#include <random>
#include <vector>

namespace nu {

class Rbf {
public:
    // inputSize  — dimensionality of each input vector x
    // numCenters — number of RBF hidden units (centers)
    // outputSize — dimensionality of the output vector y
    // lr         — SGD learning rate for output weights
    // outMode    — Linear (regression/MSE) or Softmax (classification/CE)
    // Throws std::invalid_argument if any dimension is zero.
    Rbf(size_t inputSize, size_t numCenters, size_t outputSize, double lr = 0.01,
        RnnOutput outMode = RnnOutput::Linear);

    // Sample numCenters vectors from data as RBF centers and set widths
    // heuristically: sigma = d_max / sqrt(2 * numCenters).
    // Must be called before train() or forward().
    // Throws std::invalid_argument if data is empty or has wrong input size.
    void fitCenters(const std::vector<std::vector<double>>& data);

    // Forward pass: compute output for a single input x.
    // Throws std::runtime_error if fitCenters() has not been called.
    std::vector<double> forward(const std::vector<double>& x) const;

    // Train output weights via online SGD for `epochs` passes over (inputs, targets).
    // Returns mean loss of the final epoch.
    // Throws std::runtime_error if fitCenters() has not been called.
    double train(const std::vector<std::vector<double>>& inputs,
        const std::vector<std::vector<double>>& targets, size_t epochs);

    // Reinitialise output weights (small normal random); centers/widths unchanged.
    void reshuffleWeights();

    size_t getInputSize() const noexcept { return _ni; }
    size_t getNumCenters() const noexcept { return _nc; }
    size_t getOutputSize() const noexcept { return _no; }
    double getLearningRate() const noexcept { return _lr; }
    RnnOutput getOutputMode() const noexcept { return _outMode; }
    bool isFitted() const noexcept { return _fitted; }

    void setLearningRate(double lr) noexcept { _lr = lr; }

private:
    size_t _ni, _nc, _no;
    double _lr;
    RnnOutput _outMode;
    bool _fitted = false;

    Eigen::MatrixXd _C; // [nc x ni]  center matrix
    Eigen::VectorXd _sigma; // [nc]       width per center
    Eigen::MatrixXd _Wout; // [no x nc]  output weight matrix
    Eigen::VectorXd _bout; // [no]       output bias

    std::mt19937 _rng;

    Eigen::VectorXd _hidden(const Eigen::VectorXd& x) const;
    Eigen::VectorXd _applyOutput(const Eigen::VectorXd& pre) const;
    void _initOutputWeights();

    static Eigen::VectorXd _softmax(const Eigen::VectorXd& z);
};

} // namespace nu
