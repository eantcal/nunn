//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_rbf.h"

#include <cmath>
#include <stdexcept>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

Rbf::Rbf(size_t inputSize, size_t numCenters, size_t outputSize, double lr, RnnOutput outMode)
    : _ni(inputSize)
    , _nc(numCenters)
    , _no(outputSize)
    , _lr(lr)
    , _outMode(outMode)
    , _C(Eigen::MatrixXd::Zero(static_cast<int>(numCenters), static_cast<int>(inputSize)))
    , _sigma(Eigen::VectorXd::Ones(static_cast<int>(numCenters)))
    , _Wout(Eigen::MatrixXd::Zero(static_cast<int>(outputSize), static_cast<int>(numCenters)))
    , _bout(Eigen::VectorXd::Zero(static_cast<int>(outputSize)))
    , _rng(std::random_device{}())
{
    if (inputSize == 0 || numCenters == 0 || outputSize == 0)
        throw std::invalid_argument("Rbf: all dimensions must be > 0");
    _initOutputWeights();
}

// ── Weight init ───────────────────────────────────────────────────────────────

void Rbf::_initOutputWeights()
{
    std::normal_distribution<double> d(0.0, 0.1);
    for (int i = 0; i < _Wout.rows(); ++i)
        for (int j = 0; j < _Wout.cols(); ++j)
            _Wout(i, j) = d(_rng);
    _bout.setZero();
}

// ── Center fitting ────────────────────────────────────────────────────────────

void Rbf::fitCenters(const std::vector<std::vector<double>>& data)
{
    if (data.empty())
        throw std::invalid_argument("Rbf::fitCenters: dataset is empty");
    if (data[0].size() != _ni)
        throw std::invalid_argument("Rbf::fitCenters: input size mismatch");

    std::uniform_int_distribution<size_t> pick(0, data.size() - 1);
    for (int j = 0; j < static_cast<int>(_nc); ++j) {
        const auto& s = data[pick(_rng)];
        for (int k = 0; k < static_cast<int>(_ni); ++k)
            _C(j, k) = s[static_cast<size_t>(k)];
    }

    // Heuristic width: d_max / sqrt(2 * nc), where d_max = max pairwise center distance.
    double dmax = 0.0;
    for (int a = 0; a < static_cast<int>(_nc); ++a)
        for (int b = a + 1; b < static_cast<int>(_nc); ++b)
            dmax = std::max(dmax, (_C.row(a) - _C.row(b)).norm());

    const double sig = (dmax > 0.0) ? dmax / std::sqrt(2.0 * static_cast<double>(_nc)) : 1.0;
    _sigma.setConstant(sig);
    _fitted = true;
}

// ── Forward ───────────────────────────────────────────────────────────────────

Eigen::VectorXd Rbf::_hidden(const Eigen::VectorXd& x) const
{
    Eigen::VectorXd h(static_cast<int>(_nc));
    for (int j = 0; j < static_cast<int>(_nc); ++j) {
        const double diff_sq = (_C.row(j).transpose() - x).squaredNorm();
        const double denom = 2.0 * _sigma(j) * _sigma(j);
        h(j) = std::exp(-diff_sq / (denom > 0.0 ? denom : 1e-12));
    }
    return h;
}

Eigen::VectorXd Rbf::_softmax(const Eigen::VectorXd& z)
{
    Eigen::VectorXd e = (z.array() - z.maxCoeff()).exp();
    return e / e.sum();
}

Eigen::VectorXd Rbf::_applyOutput(const Eigen::VectorXd& pre) const
{
    return (_outMode == RnnOutput::Softmax) ? _softmax(pre) : pre;
}

std::vector<double> Rbf::forward(const std::vector<double>& x) const
{
    if (!_fitted)
        throw std::runtime_error("Rbf::forward: call fitCenters() before forward()");
    if (x.size() != _ni)
        throw std::invalid_argument("Rbf::forward: input size mismatch");

    const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(x.data(), x.size());
    const Eigen::VectorXd y = _applyOutput(_Wout * _hidden(xv) + _bout);
    return std::vector<double>(y.data(), y.data() + y.size());
}

// ── Training ──────────────────────────────────────────────────────────────────

double Rbf::train(const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets, size_t epochs)
{
    if (!_fitted)
        throw std::runtime_error("Rbf::train: call fitCenters() before train()");
    if (inputs.size() != targets.size() || inputs.empty())
        throw std::invalid_argument("Rbf::train: inputs/targets size mismatch or empty");

    const int N = static_cast<int>(inputs.size());
    double lastLoss = 0.0;

    for (size_t ep = 0; ep < epochs; ++ep) {
        double total = 0.0;
        for (int s = 0; s < N; ++s) {
            const Eigen::VectorXd xv
                = Eigen::Map<const Eigen::VectorXd>(inputs[s].data(), inputs[s].size());
            const Eigen::VectorXd tv
                = Eigen::Map<const Eigen::VectorXd>(targets[s].data(), targets[s].size());

            const Eigen::VectorXd h = _hidden(xv);
            const Eigen::VectorXd pre = _Wout * h + _bout;
            const Eigen::VectorXd y = _applyOutput(pre);
            const Eigen::VectorXd delta = y - tv; // dL/d(pre) is the same for both modes

            _Wout -= _lr * (delta * h.transpose());
            _bout -= _lr * delta;

            if (_outMode == RnnOutput::Softmax)
                total += -(tv.array() * (y.array() + 1e-12).log()).sum();
            else
                total += delta.squaredNorm() / static_cast<double>(_no);
        }
        lastLoss = total / static_cast<double>(N);
    }
    return lastLoss;
}

// ── Weight reset ──────────────────────────────────────────────────────────────

void Rbf::reshuffleWeights()
{
    _initOutputWeights();
}

} // namespace nu
