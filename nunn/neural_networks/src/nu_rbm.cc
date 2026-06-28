//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_rbm.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

Rbm::Rbm(size_t nVisible, size_t nHidden, double lr, unsigned seed)
    : _nv(nVisible)
    , _nh(nHidden)
    , _lr(lr)
    , _W(Eigen::MatrixXd::Zero(nHidden, nVisible))
    , _b(Eigen::VectorXd::Zero(nVisible))
    , _c(Eigen::VectorXd::Zero(nHidden))
    , _rng(seed)
    , _udist(0.0, 1.0)
{
    if (nVisible == 0 || nHidden == 0)
        throw std::invalid_argument("Rbm: nVisible and nHidden must be > 0");

    std::normal_distribution<double> ndist(0.0, 0.01);
    for (Eigen::Index i = 0; i < _W.rows(); ++i)
        for (Eigen::Index j = 0; j < _W.cols(); ++j)
            _W(i, j) = ndist(_rng);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

Eigen::VectorXd Rbm::_sigmoid(const Eigen::VectorXd& x)
{
    return x.unaryExpr([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
}

Eigen::VectorXd Rbm::_sample(const Eigen::VectorXd& probs)
{
    Eigen::VectorXd s(probs.size());
    for (Eigen::Index i = 0; i < probs.size(); ++i)
        s(i) = (_udist(_rng) < probs(i)) ? 1.0 : 0.0;
    return s;
}

Eigen::VectorXd Rbm::_toEigen(const std::vector<double>& v) const
{
    if (v.size() != _nv)
        throw std::invalid_argument("Rbm: input size mismatch");
    return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(_nv));
}

// ── Conditional distributions ─────────────────────────────────────────────────

Eigen::VectorXd Rbm::hiddenProbs(const Eigen::VectorXd& v) const
{
    return _sigmoid(_c + _W * v);
}

Eigen::VectorXd Rbm::visibleProbs(const Eigen::VectorXd& h) const
{
    return _sigmoid(_b + _W.transpose() * h);
}

Eigen::VectorXd Rbm::sampleHidden(const Eigen::VectorXd& v)
{
    return _sample(hiddenProbs(v));
}

Eigen::VectorXd Rbm::sampleVisible(const Eigen::VectorXd& h)
{
    return _sample(visibleProbs(h));
}

// ── CD-k update ───────────────────────────────────────────────────────────────

void Rbm::trainStep(const std::vector<double>& x, size_t k)
{
    if (k == 0)
        k = 1;
    const Eigen::VectorXd v0 = _toEigen(x);

    // Positive phase: sample h0
    const Eigen::VectorXd h0_prob = hiddenProbs(v0);
    Eigen::VectorXd hk = _sample(h0_prob);

    // k Gibbs steps (last step: keep probs for h, not sample)
    Eigen::VectorXd vk;
    Eigen::VectorXd hk_prob;
    for (size_t step = 0; step < k; ++step) {
        vk = _sample(visibleProbs(hk));
        hk_prob = hiddenProbs(vk);
        hk = (step < k - 1) ? _sample(hk_prob) : hk_prob;
    }

    // Gradient ascent on log-likelihood
    _W += _lr * (h0_prob * v0.transpose() - hk_prob * vk.transpose());
    _b += _lr * (v0 - vk);
    _c += _lr * (h0_prob - hk_prob);
}

void Rbm::train(const std::vector<std::vector<double>>& dataset, size_t epochs, size_t cdK)
{
    if (dataset.empty())
        throw std::invalid_argument("Rbm::train: dataset is empty");
    if (dataset[0].size() != _nv)
        throw std::invalid_argument("Rbm::train: input size mismatch");

    std::vector<size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);

    for (size_t ep = 0; ep < epochs; ++ep) {
        std::shuffle(order.begin(), order.end(), _rng);
        for (size_t idx : order)
            trainStep(dataset[idx], cdK);
    }
}

// ── Inference ─────────────────────────────────────────────────────────────────

std::vector<double> Rbm::reconstruct(const std::vector<double>& v) const
{
    const Eigen::VectorXd ve = _toEigen(v);
    const Eigen::VectorXd h_prob = hiddenProbs(ve);
    const Eigen::VectorXd v_prob = visibleProbs(h_prob);
    return std::vector<double>(v_prob.data(), v_prob.data() + static_cast<ptrdiff_t>(_nv));
}

std::vector<double> Rbm::encode(const std::vector<double>& v) const
{
    const Eigen::VectorXd ve = _toEigen(v);
    const Eigen::VectorXd h = hiddenProbs(ve);
    return std::vector<double>(h.data(), h.data() + static_cast<ptrdiff_t>(_nh));
}

double Rbm::reconstructionError(const std::vector<std::vector<double>>& dataset) const
{
    if (dataset.empty())
        return 0.0;

    double total = 0.0;
    for (const auto& x : dataset) {
        const auto r = reconstruct(x);
        double mse = 0.0;
        for (size_t i = 0; i < _nv; ++i) {
            const double d = x[i] - r[i];
            mse += d * d;
        }
        total += mse / static_cast<double>(_nv);
    }
    return total / static_cast<double>(dataset.size());
}

} // namespace nu
