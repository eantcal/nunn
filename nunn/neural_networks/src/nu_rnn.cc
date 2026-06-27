//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_rnn.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace nu {

VanillaRnn::VanillaRnn(size_t inputSize, size_t hiddenSize, size_t outputSize, double lr,
    double gradClip, RnnOutput outMode)
    : _ni(inputSize)
    , _nh(hiddenSize)
    , _no(outputSize)
    , _lr(lr)
    , _gradClip(gradClip)
    , _outMode(outMode)
    , _Wx(Eigen::MatrixXd::Zero(hiddenSize, inputSize))
    , _Wh(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize))
    , _bh(Eigen::VectorXd::Zero(hiddenSize))
    , _Wy(Eigen::MatrixXd::Zero(outputSize, hiddenSize))
    , _by(Eigen::VectorXd::Zero(outputSize))
    , _h_prev(Eigen::VectorXd::Zero(hiddenSize))
    , _y(outputSize, 0.0)
    , _h(hiddenSize, 0.0)
{
    reshuffleWeights();
}

void VanillaRnn::resetState()
{
    _h_prev.setZero();
    std::fill(_h.begin(), _h.end(), 0.0);
    std::fill(_y.begin(), _y.end(), 0.0);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> VanillaRnn::_stepEigen(
    const Eigen::VectorXd& x, const Eigen::VectorXd& h_prev) const
{
    const Eigen::VectorXd h = (_Wx * x + _Wh * h_prev + _bh).array().tanh();
    const Eigen::VectorXd net_y = _Wy * h + _by;
    const Eigen::VectorXd y = (_outMode == RnnOutput::Softmax) ? _softmax(net_y) : net_y;
    return { h, y };
}

void VanillaRnn::step(const std::vector<double>& x)
{
    const Eigen::VectorXd xv
        = Eigen::Map<const Eigen::VectorXd>(x.data(), static_cast<Eigen::Index>(_ni));
    auto [h, y] = _stepEigen(xv, _h_prev);
    _h_prev = h;
    Eigen::Map<Eigen::VectorXd>(_y.data(), static_cast<Eigen::Index>(_no)) = y;
    Eigen::Map<Eigen::VectorXd>(_h.data(), static_cast<Eigen::Index>(_nh)) = h;
}

double VanillaRnn::bptt(const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets, size_t truncate)
{
    const size_t T = inputs.size();
    if (T == 0)
        return 0.0;

    // ── Forward pass ──────────────────────────────────────────────────────────
    // h_stored[0]   = hidden state before the sequence
    // h_stored[t+1] = hidden state after step t
    std::vector<Eigen::VectorXd> h_stored(T + 1);
    std::vector<Eigen::VectorXd> y_stored(T);
    h_stored[0] = _h_prev;

    for (size_t t = 0; t < T; ++t) {
        const Eigen::VectorXd xv
            = Eigen::Map<const Eigen::VectorXd>(inputs[t].data(), static_cast<Eigen::Index>(_ni));
        auto [h, y] = _stepEigen(xv, h_stored[t]);
        h_stored[t + 1] = std::move(h);
        y_stored[t] = std::move(y);
    }

    // ── Loss ──────────────────────────────────────────────────────────────────
    double loss = 0.0;
    for (size_t t = 0; t < T; ++t) {
        const Eigen::VectorXd tv
            = Eigen::Map<const Eigen::VectorXd>(targets[t].data(), static_cast<Eigen::Index>(_no));
        if (_outMode == RnnOutput::Softmax) {
            for (Eigen::Index k = 0; k < static_cast<Eigen::Index>(_no); ++k)
                loss -= tv[k] * std::log(std::max(y_stored[t][k], 1e-12));
        } else {
            loss += 0.5 * (y_stored[t] - tv).squaredNorm();
        }
    }
    loss /= static_cast<double>(T);

    // ── Backward pass (truncated BPTT) ────────────────────────────────────────
    Eigen::MatrixXd dWx = Eigen::MatrixXd::Zero(_nh, _ni);
    Eigen::MatrixXd dWh = Eigen::MatrixXd::Zero(_nh, _nh);
    Eigen::VectorXd dbh = Eigen::VectorXd::Zero(_nh);
    Eigen::MatrixXd dWy = Eigen::MatrixXd::Zero(_no, _nh);
    Eigen::VectorXd dby = Eigen::VectorXd::Zero(_no);

    Eigen::VectorXd dh_next = Eigen::VectorXd::Zero(_nh);

    const size_t t_start = (T > truncate) ? T - truncate : 0;
    const double inv_T = 1.0 / static_cast<double>(T);

    for (size_t t = T; t-- > t_start;) {
        const Eigen::VectorXd tv
            = Eigen::Map<const Eigen::VectorXd>(targets[t].data(), static_cast<Eigen::Index>(_no));

        // Gradient of loss w.r.t. net_y (pre-activation of output layer).
        // For both MSE+Linear and CE+Softmax this simplifies to (y - target)/T.
        const Eigen::VectorXd dy = (y_stored[t] - tv) * inv_T;

        dWy += dy * h_stored[t + 1].transpose();
        dby += dy;

        // Gradient flowing into the hidden state from output and from future
        const Eigen::VectorXd dh = _Wy.transpose() * dy + dh_next;

        // Gradient through tanh: σ'(h) = 1 − h²
        const Eigen::VectorXd dtanh
            = (1.0 - h_stored[t + 1].array().square()).matrix().cwiseProduct(dh);

        const Eigen::VectorXd xv
            = Eigen::Map<const Eigen::VectorXd>(inputs[t].data(), static_cast<Eigen::Index>(_ni));
        dWx += dtanh * xv.transpose();
        dWh += dtanh * h_stored[t].transpose(); // h_{t-1} = h_stored[t]
        dbh += dtanh;

        dh_next = _Wh.transpose() * dtanh;
    }

    // ── Gradient clipping ─────────────────────────────────────────────────────
    _clip(dWx, _gradClip);
    _clip(dWh, _gradClip);
    _clip(dbh, _gradClip);
    _clip(dWy, _gradClip);
    _clip(dby, _gradClip);

    // ── SGD weight update ─────────────────────────────────────────────────────
    _Wx -= _lr * dWx;
    _Wh -= _lr * dWh;
    _bh -= _lr * dbh;
    _Wy -= _lr * dWy;
    _by -= _lr * dby;

    // Advance hidden state to end of sequence
    _h_prev = h_stored[T];
    Eigen::Map<Eigen::VectorXd>(_h.data(), static_cast<Eigen::Index>(_nh)) = _h_prev;

    return loss;
}

void VanillaRnn::reshuffleWeights()
{
    std::mt19937 rng(std::random_device{}());

    auto initMatrix = [&](Eigen::MatrixXd& M, size_t fan_in, size_t fan_out) {
        const double scale = std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
        std::normal_distribution<double> dist(0.0, scale);
        for (Eigen::Index i = 0; i < M.rows(); ++i)
            for (Eigen::Index j = 0; j < M.cols(); ++j)
                M(i, j) = dist(rng);
    };

    initMatrix(_Wx, _ni, _nh);
    initMatrix(_Wh, _nh, _nh);
    initMatrix(_Wy, _nh, _no);
    _bh.setZero();
    _by.setZero();
    _h_prev.setZero();
    std::fill(_h.begin(), _h.end(), 0.0);
    std::fill(_y.begin(), _y.end(), 0.0);
}

Eigen::VectorXd VanillaRnn::_softmax(const Eigen::VectorXd& z)
{
    // Subtract max for numerical stability before exponentiating
    const Eigen::VectorXd shifted = z.array() - z.maxCoeff();
    const Eigen::VectorXd e = shifted.array().exp();
    return e / e.sum();
}

void VanillaRnn::_clip(Eigen::MatrixXd& m, double c)
{
    m = m.cwiseMax(-c).cwiseMin(c);
}

void VanillaRnn::_clip(Eigen::VectorXd& v, double c)
{
    v = v.cwiseMax(-c).cwiseMin(c);
}

} // namespace nu
