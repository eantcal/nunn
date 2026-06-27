//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_lstm.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

Lstm::Lstm(size_t inputSize, size_t hiddenSize, size_t outputSize, double lr, double gradClip,
    RnnOutput outMode)
    : _ni(inputSize)
    , _nh(hiddenSize)
    , _no(outputSize)
    , _lr(lr)
    , _gradClip(gradClip)
    , _outMode(outMode)
    , _W(Eigen::MatrixXd::Zero(4 * hiddenSize, inputSize))
    , _U(Eigen::MatrixXd::Zero(4 * hiddenSize, hiddenSize))
    , _b(Eigen::VectorXd::Zero(4 * hiddenSize))
    , _Wy(Eigen::MatrixXd::Zero(outputSize, hiddenSize))
    , _by(Eigen::VectorXd::Zero(outputSize))
    , _h_prev(Eigen::VectorXd::Zero(hiddenSize))
    , _c_prev(Eigen::VectorXd::Zero(hiddenSize))
    , _y(outputSize, 0.0)
    , _h(hiddenSize, 0.0)
{
    reshuffleWeights();
}

// ── State ─────────────────────────────────────────────────────────────────────

void Lstm::resetState()
{
    _h_prev.setZero();
    _c_prev.setZero();
    std::fill(_h.begin(), _h.end(), 0.0);
    std::fill(_y.begin(), _y.end(), 0.0);
}

// ── Forward step ──────────────────────────────────────────────────────────────

Lstm::StepResult Lstm::_stepEigen(
    const Eigen::VectorXd& x, const Eigen::VectorXd& h_prev, const Eigen::VectorXd& c_prev) const
{
    // Single GEMV for all four gates
    const Eigen::VectorXd pre = _W * x + _U * h_prev + _b;

    const Eigen::Index nh = static_cast<Eigen::Index>(_nh);

    StepResult r;
    r.i = _sigmoid(pre.segment(0, nh)); // input  gate
    r.f = _sigmoid(pre.segment(nh, nh)); // forget gate
    r.o = _sigmoid(pre.segment(2 * nh, nh)); // output gate
    r.g = pre.segment(3 * nh, nh).array().tanh().matrix(); // cell candidate

    r.c = r.f.cwiseProduct(c_prev) + r.i.cwiseProduct(r.g); // cell state
    r.h = r.o.cwiseProduct(r.c.array().tanh().matrix()); // hidden state

    const Eigen::VectorXd net_y = _Wy * r.h + _by;
    r.y = (_outMode == RnnOutput::Softmax) ? _softmax(net_y) : net_y;

    return r;
}

void Lstm::step(const std::vector<double>& x)
{
    const Eigen::VectorXd xv
        = Eigen::Map<const Eigen::VectorXd>(x.data(), static_cast<Eigen::Index>(_ni));
    auto r = _stepEigen(xv, _h_prev, _c_prev);
    _h_prev = r.h;
    _c_prev = r.c;
    Eigen::Map<Eigen::VectorXd>(_y.data(), static_cast<Eigen::Index>(_no)) = r.y;
    Eigen::Map<Eigen::VectorXd>(_h.data(), static_cast<Eigen::Index>(_nh)) = r.h;
}

// ── BPTT ──────────────────────────────────────────────────────────────────────

double Lstm::bptt(const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets, size_t truncate)
{
    const size_t T = inputs.size();
    if (T == 0)
        return 0.0;

    const Eigen::Index ni = static_cast<Eigen::Index>(_ni);
    const Eigen::Index nh = static_cast<Eigen::Index>(_nh);
    const Eigen::Index nh4 = 4 * nh;
    const Eigen::Index no = static_cast<Eigen::Index>(_no);

    // ── Forward pass ──────────────────────────────────────────────────────────
    // h_s[0], c_s[0]   = states before the sequence
    // h_s[t+1], c_s[t+1] = states after processing x[t]
    std::vector<Eigen::VectorXd> h_s(T + 1), c_s(T + 1);
    std::vector<Eigen::VectorXd> i_s(T), f_s(T), o_s(T), g_s(T), y_s(T);
    h_s[0] = _h_prev;
    c_s[0] = _c_prev;

    for (size_t t = 0; t < T; ++t) {
        const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(inputs[t].data(), ni);
        auto r = _stepEigen(xv, h_s[t], c_s[t]);
        h_s[t + 1] = std::move(r.h);
        c_s[t + 1] = std::move(r.c);
        i_s[t] = std::move(r.i);
        f_s[t] = std::move(r.f);
        o_s[t] = std::move(r.o);
        g_s[t] = std::move(r.g);
        y_s[t] = std::move(r.y);
    }

    // ── Loss ──────────────────────────────────────────────────────────────────
    double loss = 0.0;
    for (size_t t = 0; t < T; ++t) {
        const Eigen::VectorXd tv = Eigen::Map<const Eigen::VectorXd>(targets[t].data(), no);
        if (_outMode == RnnOutput::Softmax) {
            for (Eigen::Index k = 0; k < no; ++k)
                loss -= tv[k] * std::log(std::max(y_s[t][k], 1e-12));
        } else {
            loss += 0.5 * (y_s[t] - tv).squaredNorm();
        }
    }
    loss /= static_cast<double>(T);

    // ── Backward pass (truncated BPTT) ────────────────────────────────────────
    Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(nh4, ni);
    Eigen::MatrixXd dU = Eigen::MatrixXd::Zero(nh4, nh);
    Eigen::VectorXd db = Eigen::VectorXd::Zero(nh4);
    Eigen::MatrixXd dWy = Eigen::MatrixXd::Zero(no, nh);
    Eigen::VectorXd dby = Eigen::VectorXd::Zero(no);

    Eigen::VectorXd dh_next = Eigen::VectorXd::Zero(nh);
    Eigen::VectorXd dc_next = Eigen::VectorXd::Zero(nh);

    const size_t t_start = (T > truncate) ? T - truncate : 0;
    const double inv_T = 1.0 / static_cast<double>(T);

    for (size_t t = T; t-- > t_start;) {
        const Eigen::VectorXd tv = Eigen::Map<const Eigen::VectorXd>(targets[t].data(), no);

        // ── Output layer ──────────────────────────────────────────────────────
        // Gradient: (y - target)/T  (same form for MSE+Linear and CE+Softmax)
        const Eigen::VectorXd dy = (y_s[t] - tv) * inv_T;
        dWy += dy * h_s[t + 1].transpose();
        dby += dy;

        // Total gradient at h_{t} (from output + from future step)
        const Eigen::VectorXd dh = _Wy.transpose() * dy + dh_next;

        // ── Cell state ────────────────────────────────────────────────────────
        // h_t = o_t ⊙ tanh(c_t)
        const Eigen::VectorXd tanhc = c_s[t + 1].array().tanh().matrix();

        // dc from h path: dh ⊙ o_t ⊙ (1 − tanh²(c_t))
        const Eigen::VectorXd dc_from_h
            = (dh.array() * o_s[t].array() * (1.0 - tanhc.array().square())).matrix();
        const Eigen::VectorXd dc = dc_from_h + dc_next;

        // ── Gate gradients (post-nonlinearity) ───────────────────────────────
        // c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        const Eigen::VectorXd di = (dc.array() * g_s[t].array()).matrix();
        const Eigen::VectorXd df = (dc.array() * c_s[t].array()).matrix(); // c_{t-1}=c_s[t]
        const Eigen::VectorXd dg = (dc.array() * i_s[t].array()).matrix();
        const Eigen::VectorXd do_ = (dh.array() * tanhc.array()).matrix();

        // Propagate cell gradient to previous step
        dc_next = (dc.array() * f_s[t].array()).matrix();

        // ── Pre-activation gradients ──────────────────────────────────────────
        // sigmoid': σ'(x) = σ(x)·(1−σ(x)) = v·(1−v)
        // tanh':    tanh'(x) = 1 − tanh²(x) = 1 − g²
        const Eigen::VectorXd di_pre
            = (di.array() * i_s[t].array() * (1.0 - i_s[t].array())).matrix();
        const Eigen::VectorXd df_pre
            = (df.array() * f_s[t].array() * (1.0 - f_s[t].array())).matrix();
        const Eigen::VectorXd do_pre
            = (do_.array() * o_s[t].array() * (1.0 - o_s[t].array())).matrix();
        const Eigen::VectorXd dg_pre = (dg.array() * (1.0 - g_s[t].array().square())).matrix();

        // Stack into a single [4·nh] vector matching the gate layout [i; f; o; g]
        Eigen::VectorXd dpre(nh4);
        dpre.segment(0, nh) = di_pre;
        dpre.segment(nh, nh) = df_pre;
        dpre.segment(2 * nh, nh) = do_pre;
        dpre.segment(3 * nh, nh) = dg_pre;

        const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(inputs[t].data(), ni);
        dW += dpre * xv.transpose();
        dU += dpre * h_s[t].transpose(); // h_{t-1} = h_s[t]
        db += dpre;

        // Pass hidden gradient back in time
        dh_next = _U.transpose() * dpre;
    }

    // ── Gradient clipping ─────────────────────────────────────────────────────
    _clip(dW, _gradClip);
    _clip(dU, _gradClip);
    _clip(db, _gradClip);
    _clip(dWy, _gradClip);
    _clip(dby, _gradClip);

    // ── SGD weight update ─────────────────────────────────────────────────────
    _W -= _lr * dW;
    _U -= _lr * dU;
    _b -= _lr * db;
    _Wy -= _lr * dWy;
    _by -= _lr * dby;

    // Advance states to end of sequence
    _h_prev = h_s[T];
    _c_prev = c_s[T];
    Eigen::Map<Eigen::VectorXd>(_h.data(), nh) = _h_prev;

    return loss;
}

// ── Weight initialisation ─────────────────────────────────────────────────────

void Lstm::reshuffleWeights()
{
    std::mt19937 rng(std::random_device{}());

    auto initMatrix = [&](Eigen::MatrixXd& M, size_t fan_in, size_t fan_out) {
        const double scale = std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
        std::normal_distribution<double> dist(0.0, scale);
        for (Eigen::Index i = 0; i < M.rows(); ++i)
            for (Eigen::Index j = 0; j < M.cols(); ++j)
                M(i, j) = dist(rng);
    };

    // W has 4·nh rows; each gate block sees (ni, nh) fan
    initMatrix(_W, _ni, _nh);
    initMatrix(_U, _nh, _nh);
    initMatrix(_Wy, _nh, _no);

    _b.setZero();
    // Forget gate bias = 1: encourages the LSTM to remember at the start of
    // training, which helps gradients flow and avoids early vanishing.
    _b.segment(static_cast<Eigen::Index>(_nh), static_cast<Eigen::Index>(_nh)).setConstant(1.0);

    _by.setZero();
    _h_prev.setZero();
    _c_prev.setZero();
    std::fill(_h.begin(), _h.end(), 0.0);
    std::fill(_y.begin(), _y.end(), 0.0);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

Eigen::VectorXd Lstm::_sigmoid(const Eigen::VectorXd& z)
{
    return (1.0 / (1.0 + (-z.array()).exp())).matrix();
}

Eigen::VectorXd Lstm::_softmax(const Eigen::VectorXd& z)
{
    const Eigen::VectorXd shifted = z.array() - z.maxCoeff();
    const Eigen::VectorXd e = shifted.array().exp();
    return e / e.sum();
}

void Lstm::_clip(Eigen::MatrixXd& m, double c)
{
    m = m.cwiseMax(-c).cwiseMin(c);
}
void Lstm::_clip(Eigen::VectorXd& v, double c)
{
    v = v.cwiseMax(-c).cwiseMin(c);
}

} // namespace nu
