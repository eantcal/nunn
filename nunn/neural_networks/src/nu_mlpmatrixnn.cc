//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_mlpmatrixnn.h"
#include "nu_random_gen.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

MlpMatrixNN::MlpMatrixNN(
    const std::vector<LayerConfig>& layers, double learningRate, double momentum, CostFunction cf)
    : _inputSize(layers.empty() ? 0 : layers.front().size)
    , _lr(learningRate)
    , _momentum(momentum)
    , _cf(cf)
{
    assert(layers.size() >= 2 && "Need at least an input and an output layer");

    _input = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(_inputSize));
    _layers.reserve(layers.size() - 1);

    for (size_t i = 1; i < layers.size(); ++i) {
        const auto inSz = static_cast<Eigen::Index>(layers[i - 1].size);
        const auto outSz = static_cast<Eigen::Index>(layers[i].size);
        Layer l;
        l.W = Eigen::MatrixXd::Zero(outSz, inSz);
        l.b = Eigen::VectorXd::Zero(outSz);
        l.a = Eigen::VectorXd::Zero(outSz);
        l.delta = Eigen::VectorXd::Zero(outSz);
        l.dW = Eigen::MatrixXd::Zero(outSz, inSz);
        l.db = Eigen::VectorXd::Zero(outSz);
        l.act = layers[i].activation;
        _layers.push_back(std::move(l));
    }

    _validateCostFunction(cf, _layers.back().act);
    reshuffleWeights();
}

// ── reshuffleWeights ──────────────────────────────────────────────────────────

void MlpMatrixNN::reshuffleWeights()
{
    // Weight scale matches MlpNN: 1 / sqrt(total weight count).
    double totalW = 0.0;
    for (const auto& l : _layers)
        totalW += static_cast<double>(l.W.size());
    const double scale = std::sqrt(totalW);

    RandomGenerator<> rng;
    for (auto& l : _layers) {
        for (Eigen::Index r = 0; r < l.W.rows(); ++r)
            for (Eigen::Index c = 0; c < l.W.cols(); ++c)
                l.W(r, c) = (-1.0 + 2.0 * rng()) / scale;

        for (Eigen::Index r = 0; r < l.b.size(); ++r)
            l.b(r) = rng();

        l.dW.setZero();
        l.db.setZero();
    }
}

// ── setInputVector ────────────────────────────────────────────────────────────

void MlpMatrixNN::setInputVector(const std::vector<double>& input)
{
    assert(input.size() == _inputSize);
    _input
        = Eigen::Map<const Eigen::VectorXd>(input.data(), static_cast<Eigen::Index>(input.size()));
}

// ── feedForward ───────────────────────────────────────────────────────────────

void MlpMatrixNN::feedForward()
{
    const Eigen::VectorXd* prev = &_input;
    for (auto& l : _layers) {
        const Eigen::VectorXd z = l.W * (*prev) + l.b;
        l.a = z.unaryExpr([a = l.act](double x) { return act::forward(a, x); });
        prev = &l.a;
    }
}

// ── backPropagate ─────────────────────────────────────────────────────────────

void MlpMatrixNN::backPropagate(const std::vector<double>& target)
{
    assert(target.size() == static_cast<size_t>(_layers.back().a.size()));

    const Eigen::Map<const Eigen::VectorXd> t(
        target.data(), static_cast<Eigen::Index>(target.size()));

    // Collect the activation feeding each layer (layer 0 is fed by _input).
    std::vector<const Eigen::VectorXd*> prevA(_layers.size());
    prevA[0] = &_input;
    for (size_t i = 1; i < _layers.size(); ++i)
        prevA[i] = &_layers[i - 1].a;

    // ── Output layer delta + immediate weight update ───────────────────────────
    //
    // MSE:        δ = act'(a) ⊙ (t − a)
    // CE+Sigmoid: δ = t − a        (sigmoid derivative cancels CE gradient)
    //
    // Weights are updated immediately after the delta is computed, matching
    // MlpNN's layer-by-layer update order.  Hidden deltas are then propagated
    // through the already-updated output weights — same behaviour as MlpNN.
    //
    auto& out = _layers.back();
    if (_cf == CostFunction::CrossEntropy) {
        out.delta = t - out.a;
    } else {
        const Eigen::VectorXd d
            = out.a.unaryExpr([a = out.act](double y) { return act::backward(a, y); });
        out.delta = d.cwiseProduct(t - out.a);
    }
    const size_t outIdx = _layers.size() - 1;
    out.dW = _lr * out.delta * prevA[outIdx]->transpose() + _momentum * out.dW;
    out.db = _lr * out.delta + _momentum * out.db;
    out.W += out.dW;
    out.b += out.db;

    // ── Hidden layers: delta + update (using already-updated next W) ──────────
    //
    // δ[l] = (W[l+1]^T · δ[l+1]) ⊙ act'(a[l])
    //
    // W[l+1] is already updated at this point — intentional, mirrors MlpNN.
    //
    for (int l = static_cast<int>(_layers.size()) - 2; l >= 0; --l) {
        const auto& next = _layers[static_cast<size_t>(l + 1)];
        auto& cur = _layers[static_cast<size_t>(l)];

        const Eigen::VectorXd prop = next.W.transpose() * next.delta;
        const Eigen::VectorXd d
            = cur.a.unaryExpr([a = cur.act](double y) { return act::backward(a, y); });
        cur.delta = prop.cwiseProduct(d);

        cur.dW = _lr * cur.delta * prevA[static_cast<size_t>(l)]->transpose() + _momentum * cur.dW;
        cur.db = _lr * cur.delta + _momentum * cur.db;
        cur.W += cur.dW;
        cur.b += cur.db;
    }
}

// ── copyOutputVector ──────────────────────────────────────────────────────────

void MlpMatrixNN::copyOutputVector(std::vector<double>& out) const
{
    const auto& a = _layers.back().a;
    out.assign(a.data(), a.data() + a.size());
}

// ── getOutputSize ─────────────────────────────────────────────────────────────

size_t MlpMatrixNN::getOutputSize() const noexcept
{
    return _layers.empty() ? 0 : static_cast<size_t>(_layers.back().a.size());
}

// ── calcMSE ───────────────────────────────────────────────────────────────────

double MlpMatrixNN::calcMSE(const std::vector<double>& target) const
{
    const auto& a = _layers.back().a;
    assert(target.size() == static_cast<size_t>(a.size()));

    const Eigen::Map<const Eigen::VectorXd> t(
        target.data(), static_cast<Eigen::Index>(target.size()));

    return (a - t).squaredNorm() / static_cast<double>(a.size());
}

// ── calcCrossEntropy ──────────────────────────────────────────────────────────

double MlpMatrixNN::calcCrossEntropy(const std::vector<double>& target) const
{
    const auto& a = _layers.back().a;
    const double eps = std::numeric_limits<double>::min();
    double ce = 0.0;
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        const double y = a(i);
        const double ti = target[static_cast<size_t>(i)];
        ce -= ti * std::log(std::max(y, eps)) + (1.0 - ti) * std::log(std::max(1.0 - y, eps));
    }
    return ce / static_cast<double>(a.size());
}

} // namespace nu
