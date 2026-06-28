//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_vae.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

Vae::Vae(size_t inputDim, size_t hiddenDim, size_t latentDim, double lr, unsigned seed)
    : _nx(inputDim)
    , _nh(hiddenDim)
    , _nz(latentDim)
    , _lr(lr)
    , _W_enc(Eigen::MatrixXd::Zero(hiddenDim, inputDim))
    , _b_enc(Eigen::VectorXd::Zero(hiddenDim))
    , _W_mu(Eigen::MatrixXd::Zero(latentDim, hiddenDim))
    , _b_mu(Eigen::VectorXd::Zero(latentDim))
    , _W_lv(Eigen::MatrixXd::Zero(latentDim, hiddenDim))
    , _b_lv(Eigen::VectorXd::Zero(latentDim))
    , _W_dec(Eigen::MatrixXd::Zero(hiddenDim, latentDim))
    , _b_dec(Eigen::VectorXd::Zero(hiddenDim))
    , _W_out(Eigen::MatrixXd::Zero(inputDim, hiddenDim))
    , _b_out(Eigen::VectorXd::Zero(inputDim))
    , _rng(seed)
    , _ndist(0.0, 1.0)
{
    if (inputDim == 0 || hiddenDim == 0 || latentDim == 0)
        throw std::invalid_argument("Vae: all dimensions must be > 0");

    std::normal_distribution<double> winit(0.0, 0.1);
    auto fill = [&](Eigen::MatrixXd& W) {
        for (Eigen::Index i = 0; i < W.rows(); ++i)
            for (Eigen::Index j = 0; j < W.cols(); ++j)
                W(i, j) = winit(_rng);
    };

    fill(_W_enc);
    fill(_W_mu);
    fill(_W_lv);
    fill(_W_dec);
    fill(_W_out);
}

// ── Static helpers ────────────────────────────────────────────────────────────

Eigen::VectorXd Vae::_relu(const Eigen::VectorXd& x)
{
    return x.cwiseMax(0.0);
}

Eigen::VectorXd Vae::_sigmoid(const Eigen::VectorXd& x)
{
    return x.unaryExpr([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
}

Eigen::VectorXd Vae::_sampleNormal(size_t n)
{
    Eigen::VectorXd s(static_cast<Eigen::Index>(n));
    for (Eigen::Index i = 0; i < s.size(); ++i)
        s(i) = _ndist(_rng);
    return s;
}

Eigen::VectorXd Vae::_toEigen(const std::vector<double>& v, size_t expected) const
{
    if (v.size() != expected)
        throw std::invalid_argument("Vae: input size mismatch");
    return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<Eigen::Index>(expected));
}

// ── Decoder (shared) ──────────────────────────────────────────────────────────

Eigen::VectorXd Vae::_decodeEigen(const Eigen::VectorXd& z) const
{
    return _sigmoid(_W_out * _relu(_W_dec * z + _b_dec) + _b_out);
}

// ── Public inference ──────────────────────────────────────────────────────────

std::pair<std::vector<double>, std::vector<double>> Vae::encode(const std::vector<double>& x) const
{
    const Eigen::VectorXd xe = _toEigen(x, _nx);
    const Eigen::VectorXd h = _relu(_W_enc * xe + _b_enc);
    const Eigen::VectorXd mu = _W_mu * h + _b_mu;
    const Eigen::VectorXd lv = (_W_lv * h + _b_lv).cwiseMax(-10.0).cwiseMin(10.0);
    return { std::vector<double>(mu.data(), mu.data() + _nz),
        std::vector<double>(lv.data(), lv.data() + _nz) };
}

std::vector<double> Vae::decode(const std::vector<double>& z) const
{
    const Eigen::VectorXd ze = _toEigen(z, _nz);
    const Eigen::VectorXd r = _decodeEigen(ze);
    return std::vector<double>(r.data(), r.data() + _nx);
}

std::vector<double> Vae::reconstruct(const std::vector<double>& x) const
{
    const Eigen::VectorXd xe = _toEigen(x, _nx);
    const Eigen::VectorXd h = _relu(_W_enc * xe + _b_enc);
    const Eigen::VectorXd mu = _W_mu * h + _b_mu;
    const Eigen::VectorXd r = _decodeEigen(mu);
    return std::vector<double>(r.data(), r.data() + _nx);
}

std::vector<double> Vae::generate()
{
    const Eigen::VectorXd z = _sampleNormal(_nz);
    const Eigen::VectorXd r = _decodeEigen(z);
    return std::vector<double>(r.data(), r.data() + _nx);
}

// ── Training ──────────────────────────────────────────────────────────────────

std::pair<double, double> Vae::trainStep(const std::vector<double>& x, double klWeight)
{
    const Eigen::VectorXd xe = _toEigen(x, _nx);
    const double nxd = static_cast<double>(_nx);
    const double nzd = static_cast<double>(_nz);

    // ── Forward pass ──────────────────────────────────────────────────────────
    const Eigen::VectorXd pre_enc = _W_enc * xe + _b_enc;
    const Eigen::VectorXd h_enc = _relu(pre_enc);

    const Eigen::VectorXd mu = _W_mu * h_enc + _b_mu;
    const Eigen::VectorXd lv = (_W_lv * h_enc + _b_lv).cwiseMax(-10.0).cwiseMin(10.0);
    const Eigen::VectorXd sigma = (0.5 * lv.array()).exp();
    const Eigen::VectorXd eps = _sampleNormal(_nz);
    const Eigen::VectorXd z = mu + sigma.cwiseProduct(eps);

    const Eigen::VectorXd pre_dec = _W_dec * z + _b_dec;
    const Eigen::VectorXd h_dec = _relu(pre_dec);
    const Eigen::VectorXd pre_out = _W_out * h_dec + _b_out;
    const Eigen::VectorXd recon = _sigmoid(pre_out);

    // ── Losses ────────────────────────────────────────────────────────────────
    // Binary cross-entropy: L_recon = -mean_i(x*log(r) + (1-x)*log(1-r))
    constexpr double kEps = 1e-7;
    const double L_recon = -(xe.array() * (recon.array() + kEps).log()
        + (1.0 - xe.array()) * (1.0 - recon.array() + kEps).log())
                                .mean();
    const double L_kl = -0.5 * (1.0 + lv.array() - mu.array().square() - lv.array().exp()).mean();

    // ── Backward pass (all gradients computed before any update) ──────────────
    // BCE + sigmoid simplification: dL/d(pre_out_i) = recon_i - x_i  (no saturation)
    const Eigen::VectorXd dL_dpre_out = (recon - xe) / nxd;
    const Eigen::MatrixXd dW_out = dL_dpre_out * h_dec.transpose();
    const Eigen::VectorXd db_out = dL_dpre_out;
    const Eigen::VectorXd dL_dh_dec = _W_out.transpose() * dL_dpre_out;

    // Decoder hidden (ReLU)
    const Eigen::VectorXd dL_dpre_dec = dL_dh_dec.array() * (pre_dec.array() > 0.0).cast<double>();
    const Eigen::MatrixXd dW_dec = dL_dpre_dec * z.transpose();
    const Eigen::VectorXd db_dec = dL_dpre_dec;
    const Eigen::VectorXd dL_dz = _W_dec.transpose() * dL_dpre_dec;

    // Reparameterization + KL gradients (KL scaled by klWeight)
    // dL/dmu_j  = dL/dz_j  +  klWeight * mu_j / nz
    // dL/dlv_j  = dL/dz_j * 0.5*(z_j - mu_j)  +  klWeight * 0.5*(exp(lv_j)-1) / nz
    const Eigen::VectorXd dL_dmu = dL_dz.array() + klWeight * mu.array() / nzd;
    const Eigen::VectorXd dL_dlv = dL_dz.array() * (0.5 * (z - mu)).array()
        + klWeight * 0.5 * (lv.array().exp() - 1.0) / nzd;

    // mu and logvar projection layers
    const Eigen::MatrixXd dW_mu = dL_dmu * h_enc.transpose();
    const Eigen::VectorXd db_mu = dL_dmu;
    const Eigen::MatrixXd dW_lv = dL_dlv * h_enc.transpose();
    const Eigen::VectorXd db_lv = dL_dlv;

    // Encoder hidden (ReLU)
    const Eigen::VectorXd dL_dh_enc = _W_mu.transpose() * dL_dmu + _W_lv.transpose() * dL_dlv;
    const Eigen::VectorXd dL_dpre_enc = dL_dh_enc.array() * (pre_enc.array() > 0.0).cast<double>();
    const Eigen::MatrixXd dW_enc = dL_dpre_enc * xe.transpose();
    const Eigen::VectorXd db_enc = dL_dpre_enc;

    // ── Parameter updates ─────────────────────────────────────────────────────
    _W_out -= _lr * dW_out;
    _b_out -= _lr * db_out;
    _W_dec -= _lr * dW_dec;
    _b_dec -= _lr * db_dec;
    _W_mu -= _lr * dW_mu;
    _b_mu -= _lr * db_mu;
    _W_lv -= _lr * dW_lv;
    _b_lv -= _lr * db_lv;
    _W_enc -= _lr * dW_enc;
    _b_enc -= _lr * db_enc;

    return { L_recon, klWeight * L_kl };
}

void Vae::train(const std::vector<std::vector<double>>& dataset, size_t epochs, double warmupFrac)
{
    if (dataset.empty())
        throw std::invalid_argument("Vae::train: dataset is empty");
    if (dataset[0].size() != _nx)
        throw std::invalid_argument("Vae::train: input size mismatch");

    std::vector<size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);

    for (size_t ep = 0; ep < epochs; ++ep) {
        // Linear KL warm-up: beta goes from 0 to 1 over warmupFrac * epochs
        const double beta = (warmupFrac > 0.0 && epochs > 1)
            ? std::min(1.0, static_cast<double>(ep) / (warmupFrac * (epochs - 1)))
            : 1.0;
        std::shuffle(order.begin(), order.end(), _rng);
        for (size_t idx : order)
            trainStep(dataset[idx], beta);
    }
}

double Vae::reconstructionError(const std::vector<std::vector<double>>& dataset) const
{
    if (dataset.empty())
        return 0.0;
    constexpr double kEps = 1e-7;
    double total = 0.0;
    for (const auto& x : dataset) {
        const auto r = reconstruct(x);
        double bce = 0.0;
        for (size_t i = 0; i < _nx; ++i)
            bce -= x[i] * std::log(r[i] + kEps) + (1.0 - x[i]) * std::log(1.0 - r[i] + kEps);
        total += bce / static_cast<double>(_nx);
    }
    return total / static_cast<double>(dataset.size());
}

} // namespace nu
