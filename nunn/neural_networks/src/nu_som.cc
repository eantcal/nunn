//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_som.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

Som::Som(size_t rows, size_t cols, size_t inputDim, double lr, double initRadius, unsigned seed)
    : _rows(rows)
    , _cols(cols)
    , _dim(inputDim)
    , _lr0(lr)
    , _r0(initRadius > 0.0 ? initRadius : std::max(rows, cols) / 2.0)
    , _W(Eigen::MatrixXd::Zero(rows * cols, inputDim))
    , _pos(Eigen::MatrixXd::Zero(rows * cols, 2))
    , _rng(seed)
{
    if (rows == 0 || cols == 0 || inputDim == 0)
        throw std::invalid_argument("Som: rows, cols and inputDim must be > 0");

    // Build flat position matrix
    for (size_t r = 0; r < _rows; ++r)
        for (size_t c = 0; c < _cols; ++c) {
            const size_t i = _idx(r, c);
            _pos(i, 0) = static_cast<double>(r);
            _pos(i, 1) = static_cast<double>(c);
        }

    reshuffleWeights();
}

// ── Weight initialisation ─────────────────────────────────────────────────────

void Som::reshuffleWeights()
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (Eigen::Index i = 0; i < _W.rows(); ++i)
        for (Eigen::Index j = 0; j < _W.cols(); ++j)
            _W(i, j) = dist(_rng);
}

// ── BMU ───────────────────────────────────────────────────────────────────────

size_t Som::_bmuFlat(const Eigen::VectorXd& x) const
{
    // Squared distances from x to every neuron weight vector
    const Eigen::MatrixXd diff = _W.rowwise() - x.transpose(); // [N x d]
    const Eigen::VectorXd sq = diff.rowwise().squaredNorm(); // [N]
    Eigen::Index idx;
    sq.minCoeff(&idx);
    return static_cast<size_t>(idx);
}

std::pair<size_t, size_t> Som::bmu(const std::vector<double>& x) const
{
    if (x.size() != _dim)
        throw std::invalid_argument("Som::bmu: input size mismatch");
    const Eigen::VectorXd xe
        = Eigen::Map<const Eigen::VectorXd>(x.data(), static_cast<Eigen::Index>(_dim));
    const size_t flat = _bmuFlat(xe);
    return { flat / _cols, flat % _cols };
}

// ── Single update step ────────────────────────────────────────────────────────

void Som::update(const std::vector<double>& x, double lr, double radius)
{
    if (x.size() != _dim)
        throw std::invalid_argument("Som::update: input size mismatch");

    const Eigen::VectorXd xe
        = Eigen::Map<const Eigen::VectorXd>(x.data(), static_cast<Eigen::Index>(_dim));
    const size_t bmu_i = _bmuFlat(xe);

    const Eigen::VectorXd bmu_pos = _pos.row(bmu_i); // [2]
    const double two_r2 = 2.0 * radius * radius;

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(_rows * _cols); ++i) {
        const double d2 = (_pos.row(i) - bmu_pos.transpose()).squaredNorm();
        const double h = std::exp(-d2 / two_r2);
        _W.row(i) += lr * h * (xe.transpose() - _W.row(i));
    }
}

// ── Full training ─────────────────────────────────────────────────────────────

void Som::train(const std::vector<std::vector<double>>& dataset, size_t epochs, double finalLr,
    double finalRadius)
{
    if (dataset.empty())
        throw std::invalid_argument("Som::train: dataset is empty");
    if (dataset[0].size() != _dim)
        throw std::invalid_argument("Som::train: input size mismatch");
    if (epochs == 0)
        return;

    // Exponential decay constants: val(t) = val_0 * exp(-t * lambda)
    // val(T-1) = finalVal  =>  lambda = -log(finalVal / val_0) / (T-1)
    const double eps = 1e-12;
    const double T = static_cast<double>(epochs);

    const double lrLambda
        = (epochs > 1) ? -std::log(std::max(finalLr, eps) / std::max(_lr0, eps)) / (T - 1.0) : 0.0;
    const double rLambda = (epochs > 1)
        ? -std::log(std::max(finalRadius, eps) / std::max(_r0, eps)) / (T - 1.0)
        : 0.0;

    // Build shuffled index vector
    std::vector<size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);

    for (size_t ep = 0; ep < epochs; ++ep) {
        const double t = static_cast<double>(ep);
        const double lr = _lr0 * std::exp(-lrLambda * t);
        const double r = _r0 * std::exp(-rLambda * t);

        std::shuffle(order.begin(), order.end(), _rng);
        for (size_t idx : order)
            update(dataset[idx], lr, r);
    }
}

// ── Quantization error ────────────────────────────────────────────────────────

double Som::quantizationError(const std::vector<std::vector<double>>& dataset) const
{
    if (dataset.empty())
        return 0.0;

    double total = 0.0;
    for (const auto& x : dataset) {
        const Eigen::VectorXd xe
            = Eigen::Map<const Eigen::VectorXd>(x.data(), static_cast<Eigen::Index>(_dim));
        const size_t b = _bmuFlat(xe);
        total += (xe - _W.row(b).transpose()).norm();
    }
    return total / static_cast<double>(dataset.size());
}

// ── Weight access ─────────────────────────────────────────────────────────────

std::vector<double> Som::getWeights(size_t r, size_t c) const
{
    if (r >= _rows || c >= _cols)
        throw std::out_of_range("Som::getWeights: index out of range");
    const Eigen::VectorXd w = _W.row(static_cast<Eigen::Index>(_idx(r, c)));
    return std::vector<double>(w.data(), w.data() + _dim);
}

} // namespace nu
