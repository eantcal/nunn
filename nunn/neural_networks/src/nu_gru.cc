//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_gru.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

Gru::Gru(size_t inputSize, size_t hiddenSize, size_t outputSize, double lr, double gradClip,
    RnnOutput outMode)
    : _ni(inputSize)
    , _nh(hiddenSize)
    , _no(outputSize)
    , _lr(lr)
    , _gradClip(gradClip)
    , _outMode(outMode)
    , _W(Eigen::MatrixXd::Zero(3 * hiddenSize, inputSize))
    , _Urz(Eigen::MatrixXd::Zero(2 * hiddenSize, hiddenSize))
    , _Uh(Eigen::MatrixXd::Zero(hiddenSize, hiddenSize))
    , _b(Eigen::VectorXd::Zero(3 * hiddenSize))
    , _Wy(Eigen::MatrixXd::Zero(outputSize, hiddenSize))
    , _by(Eigen::VectorXd::Zero(outputSize))
    , _h_prev(Eigen::VectorXd::Zero(hiddenSize))
    , _y(outputSize, 0.0)
    , _h(hiddenSize, 0.0)
{
    reshuffleWeights();
}

// ── State ─────────────────────────────────────────────────────────────────────

void Gru::resetState()
{
    _h_prev.setZero();
    std::fill(_h.begin(), _h.end(), 0.0);
    std::fill(_y.begin(), _y.end(), 0.0);
}

// ── Forward step ──────────────────────────────────────────────────────────────

Gru::StepResult Gru::_stepEigen(const Eigen::VectorXd& x, const Eigen::VectorXd& h_prev) const
{
    const Eigen::Index nh = static_cast<Eigen::Index>(_nh);
    const Eigen::Index nh2 = 2 * nh;

    // Input projection for all three gates in one GEMV
    const Eigen::VectorXd Wx = _W * x; // [3·nh]

    // r and z: can share a single recurrent GEMV
    const Eigen::VectorXd pre_rz = Wx.head(nh2) + _Urz * h_prev + _b.head(nh2);

    StepResult r;
    r.r = _sigmoid(pre_rz.head(nh)); // reset gate
    r.z = _sigmoid(pre_rz.tail(nh)); // update gate

    r.rh = r.r.cwiseProduct(h_prev); // r ⊙ h_{t-1}

    const Eigen::VectorXd pre_g = Wx.tail(nh) + _Uh * r.rh + _b.tail(nh);
    r.g = pre_g.array().tanh().matrix(); // candidate

    r.h = (1.0 - r.z.array()).matrix().cwiseProduct(h_prev) + r.z.cwiseProduct(r.g); // new hidden

    const Eigen::VectorXd net_y = _Wy * r.h + _by;
    r.y = (_outMode == RnnOutput::Softmax) ? _softmax(net_y) : net_y;

    return r;
}

void Gru::step(const std::vector<double>& x)
{
    const Eigen::VectorXd xv
        = Eigen::Map<const Eigen::VectorXd>(x.data(), static_cast<Eigen::Index>(_ni));
    auto r = _stepEigen(xv, _h_prev);
    _h_prev = r.h;
    Eigen::Map<Eigen::VectorXd>(_y.data(), static_cast<Eigen::Index>(_no)) = r.y;
    Eigen::Map<Eigen::VectorXd>(_h.data(), static_cast<Eigen::Index>(_nh)) = r.h;
}

// ── BPTT ──────────────────────────────────────────────────────────────────────

double Gru::bptt(const std::vector<std::vector<double>>& inputs,
    const std::vector<std::vector<double>>& targets, size_t truncate)
{
    const size_t T = inputs.size();
    if (T == 0)
        return 0.0;

    const Eigen::Index ni = static_cast<Eigen::Index>(_ni);
    const Eigen::Index nh = static_cast<Eigen::Index>(_nh);
    const Eigen::Index nh2 = 2 * nh;
    const Eigen::Index nh3 = 3 * nh;
    const Eigen::Index no = static_cast<Eigen::Index>(_no);

    // ── Forward pass ──────────────────────────────────────────────────────────
    // h_s[0] = state before sequence; h_s[t+1] = state after step t
    std::vector<Eigen::VectorXd> h_s(T + 1);
    std::vector<Eigen::VectorXd> r_s(T), z_s(T), g_s(T), rh_s(T), y_s(T);
    h_s[0] = _h_prev;

    for (size_t t = 0; t < T; ++t) {
        const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(inputs[t].data(), ni);
        auto res = _stepEigen(xv, h_s[t]);
        h_s[t + 1] = std::move(res.h);
        r_s[t] = std::move(res.r);
        z_s[t] = std::move(res.z);
        g_s[t] = std::move(res.g);
        rh_s[t] = std::move(res.rh);
        y_s[t] = std::move(res.y);
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
    Eigen::MatrixXd dW = Eigen::MatrixXd::Zero(nh3, ni);
    Eigen::MatrixXd dUrz = Eigen::MatrixXd::Zero(nh2, nh);
    Eigen::MatrixXd dUh = Eigen::MatrixXd::Zero(nh, nh);
    Eigen::VectorXd db = Eigen::VectorXd::Zero(nh3);
    Eigen::MatrixXd dWy = Eigen::MatrixXd::Zero(no, nh);
    Eigen::VectorXd dby = Eigen::VectorXd::Zero(no);

    Eigen::VectorXd dh_next = Eigen::VectorXd::Zero(nh);

    const size_t t_start = (T > truncate) ? T - truncate : 0;
    const double inv_T = 1.0 / static_cast<double>(T);

    for (size_t t = T; t-- > t_start;) {
        const Eigen::VectorXd tv = Eigen::Map<const Eigen::VectorXd>(targets[t].data(), no);

        // ── Output layer ──────────────────────────────────────────────────────
        const Eigen::VectorXd dy = (y_s[t] - tv) * inv_T;
        dWy += dy * h_s[t + 1].transpose();
        dby += dy;

        const Eigen::VectorXd dh = _Wy.transpose() * dy + dh_next;

        // ── h = (1−z)⊙h_prev + z⊙g ──────────────────────────────────────────
        const Eigen::VectorXd dz = dh.cwiseProduct(g_s[t] - h_s[t]); // h_s[t] = h_{t-1}
        const Eigen::VectorXd dg = dh.cwiseProduct(z_s[t]);
        const Eigen::VectorXd dh_prev_1 = dh.cwiseProduct((1.0 - z_s[t].array()).matrix());

        // ── g = tanh(Wh·x + Uh·rh + bh) ─────────────────────────────────────
        const Eigen::VectorXd dpre_g = dg.cwiseProduct((1.0 - g_s[t].array().square()).matrix());
        const Eigen::VectorXd d_rh = _Uh.transpose() * dpre_g; // grad w.r.t. r⊙h_prev
        const Eigen::VectorXd dr = d_rh.cwiseProduct(h_s[t]);
        const Eigen::VectorXd dh_prev_2 = d_rh.cwiseProduct(r_s[t]);

        // ── z = σ(pre_z),  r = σ(pre_r) ──────────────────────────────────────
        const Eigen::VectorXd dpre_z
            = (dz.array() * z_s[t].array() * (1.0 - z_s[t].array())).matrix();
        const Eigen::VectorXd dpre_r
            = (dr.array() * r_s[t].array() * (1.0 - r_s[t].array())).matrix();

        // Stack [dpre_r; dpre_z] for the joint rz recurrent weight
        Eigen::VectorXd dpre_rz(nh2);
        dpre_rz.head(nh) = dpre_r;
        dpre_rz.tail(nh) = dpre_z;

        const Eigen::VectorXd dh_prev_3 = _Urz.transpose() * dpre_rz;

        // ── Propagate hidden gradient to previous step ────────────────────────
        dh_next = dh_prev_1 + dh_prev_2 + dh_prev_3;

        // ── Accumulate weight gradients ───────────────────────────────────────
        const Eigen::VectorXd xv = Eigen::Map<const Eigen::VectorXd>(inputs[t].data(), ni);
        dW.topRows(nh2) += dpre_rz * xv.transpose();
        dW.bottomRows(nh) += dpre_g * xv.transpose();
        dUrz += dpre_rz * h_s[t].transpose();
        dUh += dpre_g * rh_s[t].transpose();
        db.head(nh2) += dpre_rz;
        db.tail(nh) += dpre_g;
    }

    // ── Gradient clipping ─────────────────────────────────────────────────────
    _clip(dW, _gradClip);
    _clip(dUrz, _gradClip);
    _clip(dUh, _gradClip);
    _clip(db, _gradClip);
    _clip(dWy, _gradClip);
    _clip(dby, _gradClip);

    // ── SGD weight update ─────────────────────────────────────────────────────
    _W -= _lr * dW;
    _Urz -= _lr * dUrz;
    _Uh -= _lr * dUh;
    _b -= _lr * db;
    _Wy -= _lr * dWy;
    _by -= _lr * dby;

    _h_prev = h_s[T];
    Eigen::Map<Eigen::VectorXd>(_h.data(), nh) = _h_prev;

    return loss;
}

// ── Weight initialisation ─────────────────────────────────────────────────────

void Gru::reshuffleWeights()
{
    std::mt19937 rng(std::random_device{}());

    auto initMatrix = [&](Eigen::MatrixXd& M, size_t fan_in, size_t fan_out) {
        const double scale = std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
        std::normal_distribution<double> dist(0.0, scale);
        for (Eigen::Index i = 0; i < M.rows(); ++i)
            for (Eigen::Index j = 0; j < M.cols(); ++j)
                M(i, j) = dist(rng);
    };

    initMatrix(_W, _ni, _nh);
    initMatrix(_Urz, _nh, _nh);
    initMatrix(_Uh, _nh, _nh);
    initMatrix(_Wy, _nh, _no);

    _b.setZero();
    _by.setZero();
    _h_prev.setZero();
    std::fill(_h.begin(), _h.end(), 0.0);
    std::fill(_y.begin(), _y.end(), 0.0);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

Eigen::VectorXd Gru::_sigmoid(const Eigen::VectorXd& z)
{
    return (1.0 / (1.0 + (-z.array()).exp())).matrix();
}

Eigen::VectorXd Gru::_softmax(const Eigen::VectorXd& z)
{
    const Eigen::VectorXd shifted = z.array() - z.maxCoeff();
    const Eigen::VectorXd e = shifted.array().exp();
    return e / e.sum();
}

void Gru::_clip(Eigen::MatrixXd& m, double c)
{
    m = m.cwiseMax(-c).cwiseMin(c);
}
void Gru::_clip(Eigen::VectorXd& v, double c)
{
    v = v.cwiseMax(-c).cwiseMin(c);
}

} // namespace nu
