//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Restricted Boltzmann Machine (RBM) with Contrastive Divergence training.
//
// Architecture:
//   Bipartite graph: nVisible binary visible units  v in {0,1}^n
//                    nHidden  binary hidden  units  h in {0,1}^m
//
// Energy:
//   E(v,h) = -b^T v - c^T h - h^T W v
//   W [m x n], b [n] visible bias, c [m] hidden bias.
//
// Conditional distributions (sigmoid):
//   P(h_j=1|v) = sigma( c_j + W_j. v )
//   P(v_i=1|h) = sigma( b_i + W._i h )
//
// Training: Contrastive Divergence CD-k (online, one sample at a time)
//   Positive phase: h0 ~ P(h|v0)
//   k Gibbs steps:  vk ~ P(v|h_{k-1}), hk ~ P(h|vk)   [last step: use probs]
//   DeltaW = lr * (h0_prob * v0^T - hk_prob * vk^T)
//   Deltab  = lr * (v0 - vk)
//   Deltac  = lr * (h0_prob - hk_prob)
//
// Internal storage:
//   _W [nh x nv]   weight matrix
//   _b [nv]        visible bias
//   _c [nh]        hidden bias
//

#pragma once

#include <Eigen/Core>
#include <random>
#include <vector>

namespace nu {

class Rbm {
public:
    // nVisible -- number of visible (input) units
    // nHidden  -- number of hidden units
    // lr       -- learning rate; default 0.01
    // seed     -- RNG seed for weight init and Gibbs sampling
    // Throws std::invalid_argument if any dimension is zero.
    Rbm(size_t nVisible, size_t nHidden, double lr = 0.01, unsigned seed = 42);

    // P(h_j=1|v) for all j: sigma(c + W*v)   [nHidden]
    Eigen::VectorXd hiddenProbs(const Eigen::VectorXd& v) const;

    // P(v_i=1|h) for all i: sigma(b + W^T*h) [nVisible]
    Eigen::VectorXd visibleProbs(const Eigen::VectorXd& h) const;

    // Sample binary hidden state given visible.
    Eigen::VectorXd sampleHidden(const Eigen::VectorXd& v);

    // Sample binary visible state given hidden.
    Eigen::VectorXd sampleVisible(const Eigen::VectorXd& h);

    // CD-k update on a single sample.
    // Throws std::invalid_argument if x.size() != nVisible().
    void trainStep(const std::vector<double>& x, size_t k = 1);

    // Full online CD-k training: dataset shuffled each epoch.
    // Throws std::invalid_argument if dataset is empty or size mismatches.
    void train(const std::vector<std::vector<double>>& dataset, size_t epochs, size_t cdK = 1);

    // Soft reconstruction v -> P(h|v) -> P(v|h).  No sampling; returns probs.
    // Throws std::invalid_argument if v.size() != nVisible().
    std::vector<double> reconstruct(const std::vector<double>& v) const;

    // Encode: returns P(h_j=1|v) as the hidden representation [nHidden].
    std::vector<double> encode(const std::vector<double>& v) const;

    // Mean MSE between each input and its soft reconstruction.
    double reconstructionError(const std::vector<std::vector<double>>& dataset) const;

    size_t nVisible() const noexcept { return _nv; }
    size_t nHidden() const noexcept { return _nh; }

private:
    size_t _nv, _nh;
    double _lr;
    Eigen::MatrixXd _W; // [nh x nv]
    Eigen::VectorXd _b; // visible bias [nv]
    Eigen::VectorXd _c; // hidden bias  [nh]
    std::mt19937 _rng;
    std::uniform_real_distribution<double> _udist;

    static Eigen::VectorXd _sigmoid(const Eigen::VectorXd& x);
    Eigen::VectorXd _sample(const Eigen::VectorXd& probs);
    Eigen::VectorXd _toEigen(const std::vector<double>& v) const;
};

} // namespace nu
