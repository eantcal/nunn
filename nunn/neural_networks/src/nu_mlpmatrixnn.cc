//
// This file is part of the nunn Library
// Copyright (c) 2026 Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

// Must precede any Windows or ArrayFire includes to suppress min/max macros.
#define NOMINMAX

#include "nu_mlpmatrixnn.h"
#include "nu_activation.h"
#include "nu_random_gen.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

// ── ArrayFire activation helpers (compiled only when NUNN_HAS_ARRAYFIRE) ──────

#ifdef NUNN_HAS_ARRAYFIRE
namespace {

void initialize_opencl_backend()
{
    af::setBackend(AF_BACKEND_OPENCL);
    af::setDevice(0);
    af::sync();
}

af::array af_activate(nu::Activation act, const af::array& z)
{
    switch (act) {
    case nu::Activation::Sigmoid:
        return 1.0 / (1.0 + af::exp(-z));
    case nu::Activation::Tanh:
        return af::tanh(z);
    case nu::Activation::ReLU:
        return af::select(z > 0.0, z, af::constant(0.0, z.dims(), f64));
    case nu::Activation::LeakyReLU:
        return af::select(z > 0.0, z, nu::act::LEAKY_RELU_ALPHA * z);
    case nu::Activation::Linear:
        return z;
    }
    return z;
}

// Derivative of activation given the activation output a = f(z).
af::array af_activate_backward(nu::Activation act, const af::array& a)
{
    switch (act) {
    case nu::Activation::Sigmoid:
        return a * (1.0 - a);
    case nu::Activation::Tanh:
        return 1.0 - a * a;
    case nu::Activation::ReLU:
        return (a > 0.0).as(f64);
    case nu::Activation::LeakyReLU:
        return af::select(a > 0.0, af::constant(1.0, a.dims(), f64),
            af::constant(nu::act::LEAKY_RELU_ALPHA, a.dims(), f64));
    case nu::Activation::Linear:
        return af::constant(1.0, a.dims(), f64);
    }
    return af::constant(1.0, a.dims(), f64);
}

} // anonymous namespace
#endif // NUNN_HAS_ARRAYFIRE

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

MlpMatrixNN::MlpMatrixNN(const std::vector<LayerConfig>& layers, double learningRate,
    double momentum, CostFunction cf, ComputeBackend backend)
    : _inputSize(layers.empty() ? 0 : layers.front().size)
    , _lr(learningRate)
    , _momentum(momentum)
    , _cf(cf)
    , _backend(backend)
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
        l.mW = Eigen::MatrixXd::Zero(outSz, inSz);
        l.vW = Eigen::MatrixXd::Zero(outSz, inSz);
        l.mb = Eigen::VectorXd::Zero(outSz);
        l.vb = Eigen::VectorXd::Zero(outSz);
        l.act = layers[i].activation;
        _layers.push_back(std::move(l));
    }

    _validateCostFunction(cf, _layers.back().act);

#ifndef NUNN_HAS_ARRAYFIRE
    if (_backend == ComputeBackend::OpenCL)
        throw std::runtime_error("MlpMatrixNN: ArrayFire/OpenCL backend not available; "
                                 "rebuild with NUNN_HAS_ARRAYFIRE defined");
    _backend = ComputeBackend::Eigen;
#else
    const bool requestedAuto = (_backend == ComputeBackend::Auto);
    if (_backend == ComputeBackend::Auto || _backend == ComputeBackend::OpenCL) {
        try {
            initialize_opencl_backend();
            _backend = ComputeBackend::OpenCL;
        } catch (const std::exception& e) {
            if (!requestedAuto)
                throw std::runtime_error(
                    std::string("MlpMatrixNN: cannot initialize ArrayFire/OpenCL backend: ")
                    + e.what());
            _backend = ComputeBackend::Eigen;
        }
    }
#endif

    try {
        reshuffleWeights();
    } catch (const std::exception&) {
#ifdef NUNN_HAS_ARRAYFIRE
        if (backend == ComputeBackend::Auto && _backend == ComputeBackend::OpenCL) {
            _backend = ComputeBackend::Eigen;
            reshuffleWeights();
        } else
#endif
        {
            throw;
        }
    }
}

// ── reshuffleWeights ──────────────────────────────────────────────────────────

void MlpMatrixNN::reshuffleWeights()
{
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
        l.mW.setZero();
        l.vW.setZero();
        l.mb.setZero();
        l.vb.setZero();
    }
    _adamT = 0;

#ifdef NUNN_HAS_ARRAYFIRE
    if (_backend == ComputeBackend::OpenCL) {
        for (auto& l : _layers) {
            // Eigen MatrixXd and ArrayFire both use column-major order, so
            // the raw data pointer can be uploaded directly.
            l.W_af = af::array(l.W.rows(), l.W.cols(), l.W.data(), afHost);
            l.b_af = af::array(static_cast<dim_t>(l.b.size()), (dim_t)1, l.b.data(), afHost);
            l.dW_af = af::constant(0.0, l.W.rows(), l.W.cols(), f64);
            l.db_af = af::constant(0.0, static_cast<dim_t>(l.b.size()), (dim_t)1, f64);
            l.mW_af.reset();
            l.vW_af.reset();
            l.mb_af.reset();
            l.vb_af.reset();
        }
    }
#endif
}

// ── setOptimizer ──────────────────────────────────────────────────────────────

void MlpMatrixNN::setOptimizer(Optimizer opt, double beta1, double beta2, double eps) noexcept
{
    _optimizer = opt;
    _beta1 = beta1;
    _beta2 = beta2;
    _adamEps = eps;
    // Reset Adam state so moments from a previous run don't bleed in.
    for (auto& l : _layers) {
        l.mW.setZero();
        l.vW.setZero();
        l.mb.setZero();
        l.vb.setZero();
#ifdef NUNN_HAS_ARRAYFIRE
        l.mW_af.reset();
        l.vW_af.reset();
        l.mb_af.reset();
        l.vb_af.reset();
#endif
    }
    _adamT = 0;
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
    if (_backend == ComputeBackend::Eigen) {
        const Eigen::VectorXd* prev = &_input;
        for (auto& l : _layers) {
            const Eigen::VectorXd z = l.W * (*prev) + l.b;
            l.a = z.unaryExpr([a = l.act](double x) { return act::forward(a, x); });
            prev = &l.a;
        }
        return;
    }

#ifdef NUNN_HAS_ARRAYFIRE
    af::array x = af::array(static_cast<dim_t>(_inputSize), (dim_t)1, _input.data(), afHost);
    for (auto& l : _layers) {
        const af::array z = af::matmul(*l.W_af, x) + *l.b_af;
        l.a_af = af_activate(l.act, z);
        x = *l.a_af;
    }
    // Sync output layer to host so copyOutputVector / calcMSE work correctly.
    _layers.back().a_af->host(_layers.back().a.data());
#endif
}

// ── backPropagate ─────────────────────────────────────────────────────────────

void MlpMatrixNN::backPropagate(const std::vector<double>& target)
{
    assert(target.size() == static_cast<size_t>(_layers.back().a.size()));

    if (_backend == ComputeBackend::Eigen) {
        const Eigen::Map<const Eigen::VectorXd> t(
            target.data(), static_cast<Eigen::Index>(target.size()));

        // Collect prevA pointers (layer 0 is fed by _input).
        std::vector<const Eigen::VectorXd*> prevA(_layers.size());
        prevA[0] = &_input;
        for (size_t i = 1; i < _layers.size(); ++i)
            prevA[i] = &_layers[i - 1].a;

        // Output layer: delta + immediate weight update.
        // MSE:        δ = act'(a) ⊙ (t − a)
        // CE+Sigmoid: δ = t − a  (sigmoid derivative cancels CE gradient)
        auto& out = _layers.back();
        if (_cf == CostFunction::CrossEntropy) {
            out.delta = t - out.a;
        } else {
            const Eigen::VectorXd d
                = out.a.unaryExpr([a = out.act](double y) { return act::backward(a, y); });
            out.delta = d.cwiseProduct(t - out.a);
        }
        const size_t outIdx = _layers.size() - 1;

        if (_optimizer == Optimizer::Adam) {
            // Adam: compute all deltas with original weights, then update all layers.
            for (int l = static_cast<int>(_layers.size()) - 2; l >= 0; --l) {
                const auto& next = _layers[static_cast<size_t>(l + 1)];
                auto& cur = _layers[static_cast<size_t>(l)];
                const Eigen::VectorXd prop = next.W.transpose() * next.delta;
                const Eigen::VectorXd d
                    = cur.a.unaryExpr([a = cur.act](double y) { return act::backward(a, y); });
                cur.delta = prop.cwiseProduct(d);
            }
            ++_adamT;
            const double bc1 = 1.0 - std::pow(_beta1, static_cast<double>(_adamT));
            const double bc2 = 1.0 - std::pow(_beta2, static_cast<double>(_adamT));
            for (size_t l = 0; l < _layers.size(); ++l) {
                auto& lay = _layers[l];
                const Eigen::MatrixXd gW = lay.delta * prevA[l]->transpose();
                const Eigen::VectorXd gb = lay.delta;
                lay.mW = _beta1 * lay.mW + (1.0 - _beta1) * gW;
                lay.vW = _beta2 * lay.vW + (1.0 - _beta2) * gW.cwiseProduct(gW);
                lay.mb = _beta1 * lay.mb + (1.0 - _beta1) * gb;
                lay.vb = _beta2 * lay.vb + (1.0 - _beta2) * gb.cwiseProduct(gb);
                const Eigen::MatrixXd mWhat = lay.mW / bc1;
                const Eigen::MatrixXd vWhat = lay.vW / bc2;
                const Eigen::VectorXd mbhat = lay.mb / bc1;
                const Eigen::VectorXd vbhat = lay.vb / bc2;
                lay.W += _lr
                    * mWhat.cwiseQuotient(vWhat.cwiseSqrt()
                        + Eigen::MatrixXd::Constant(vWhat.rows(), vWhat.cols(), _adamEps));
                lay.b += _lr
                    * mbhat.cwiseQuotient(
                        vbhat.cwiseSqrt() + Eigen::VectorXd::Constant(vbhat.size(), _adamEps));
            }
        } else {
            // SGD + momentum: update output layer first, then propagate delta through
            // already-updated weights (mirrors MlpNN's immediate-update order).
            out.dW = _lr * out.delta * prevA[outIdx]->transpose() + _momentum * out.dW;
            out.db = _lr * out.delta + _momentum * out.db;
            out.W += out.dW;
            out.b += out.db;
            for (int l = static_cast<int>(_layers.size()) - 2; l >= 0; --l) {
                const auto& next = _layers[static_cast<size_t>(l + 1)];
                auto& cur = _layers[static_cast<size_t>(l)];
                const Eigen::VectorXd prop = next.W.transpose() * next.delta;
                const Eigen::VectorXd d
                    = cur.a.unaryExpr([a = cur.act](double y) { return act::backward(a, y); });
                cur.delta = prop.cwiseProduct(d);
                cur.dW = _lr * cur.delta * prevA[static_cast<size_t>(l)]->transpose()
                    + _momentum * cur.dW;
                cur.db = _lr * cur.delta + _momentum * cur.db;
                cur.W += cur.dW;
                cur.b += cur.db;
            }
        }
        return;
    }

#ifdef NUNN_HAS_ARRAYFIRE
    const dim_t inSz = static_cast<dim_t>(_inputSize);
    const dim_t ouSz = static_cast<dim_t>(_layers.back().a.size());

    af::array x0 = af::array(inSz, (dim_t)1, _input.data(), afHost);
    af::array t = af::array(ouSz, (dim_t)1, target.data(), afHost);

    // Output layer delta.
    auto& out = _layers.back();
    if (_cf == CostFunction::CrossEntropy)
        out.delta_af = t - *out.a_af;
    else
        out.delta_af = af_activate_backward(out.act, *out.a_af) * (t - *out.a_af);

    const size_t outIdx = _layers.size() - 1;

    if (_optimizer == Optimizer::Adam) {
        // Adam must derive every delta from the same pre-update parameters.
        for (int l = static_cast<int>(_layers.size()) - 2; l >= 0; --l) {
            const size_t lu = static_cast<size_t>(l);
            auto& next = _layers[lu + 1];
            auto& cur = _layers[lu];
            const af::array prop
                = af::matmul(*next.W_af, *next.delta_af, AF_MAT_TRANS, AF_MAT_NONE);
            cur.delta_af = prop * af_activate_backward(cur.act, *cur.a_af);
        }

        ++_adamT;
        const double bc1 = 1.0 - std::pow(_beta1, static_cast<double>(_adamT));
        const double bc2 = 1.0 - std::pow(_beta2, static_cast<double>(_adamT));
        for (size_t l = 0; l < _layers.size(); ++l) {
            auto& lay = _layers[l];
            const af::array& prevA = (l == 0) ? x0 : *_layers[l - 1].a_af;
            const af::array gW = af::matmul(*lay.delta_af, prevA, AF_MAT_NONE, AF_MAT_TRANS);
            const af::array gb = *lay.delta_af;

            if (!lay.mW_af) {
                lay.mW_af = af::constant(0.0, lay.W.rows(), lay.W.cols(), f64);
                lay.vW_af = af::constant(0.0, lay.W.rows(), lay.W.cols(), f64);
                lay.mb_af = af::constant(0.0, static_cast<dim_t>(lay.b.size()), (dim_t)1, f64);
                lay.vb_af = af::constant(0.0, static_cast<dim_t>(lay.b.size()), (dim_t)1, f64);
            }

            *lay.mW_af = _beta1 * *lay.mW_af + (1.0 - _beta1) * gW;
            *lay.vW_af = _beta2 * *lay.vW_af + (1.0 - _beta2) * gW * gW;
            *lay.mb_af = _beta1 * *lay.mb_af + (1.0 - _beta1) * gb;
            *lay.vb_af = _beta2 * *lay.vb_af + (1.0 - _beta2) * gb * gb;

            const af::array mWhat = *lay.mW_af / bc1;
            const af::array vWhat = *lay.vW_af / bc2;
            const af::array mbhat = *lay.mb_af / bc1;
            const af::array vbhat = *lay.vb_af / bc2;
            *lay.W_af += _lr * mWhat / (af::sqrt(vWhat) + _adamEps);
            *lay.b_af += _lr * mbhat / (af::sqrt(vbhat) + _adamEps);
        }
    } else {
        // SGD + momentum mirrors the Eigen path's immediate-update order.
        const af::array& prevAOut = (outIdx == 0) ? x0 : *_layers[outIdx - 1].a_af;
        out.dW_af = _lr * af::matmul(*out.delta_af, prevAOut, AF_MAT_NONE, AF_MAT_TRANS)
            + _momentum * *out.dW_af;
        out.db_af = _lr * *out.delta_af + _momentum * *out.db_af;
        *out.W_af += *out.dW_af;
        *out.b_af += *out.db_af;

        for (int l = static_cast<int>(_layers.size()) - 2; l >= 0; --l) {
            const size_t lu = static_cast<size_t>(l);
            auto& next = _layers[lu + 1];
            auto& cur = _layers[lu];
            const af::array& prevA = (lu == 0) ? x0 : *_layers[lu - 1].a_af;
            const af::array prop
                = af::matmul(*next.W_af, *next.delta_af, AF_MAT_TRANS, AF_MAT_NONE);
            cur.delta_af = prop * af_activate_backward(cur.act, *cur.a_af);
            cur.dW_af = _lr * af::matmul(*cur.delta_af, prevA, AF_MAT_NONE, AF_MAT_TRANS)
                + _momentum * *cur.dW_af;
            cur.db_af = _lr * *cur.delta_af + _momentum * *cur.db_af;
            *cur.W_af += *cur.dW_af;
            *cur.b_af += *cur.db_af;
        }
    }
#endif
}

// ── trainBatch ────────────────────────────────────────────────────────────────

void MlpMatrixNN::trainBatch(
    const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets)
{
    if (inputs.empty() || inputs.size() != targets.size())
        throw std::invalid_argument(
            "trainBatch: batch must be non-empty and inputs/targets must have the same size");

    if (_backend == ComputeBackend::Eigen) {
        const auto B = static_cast<Eigen::Index>(inputs.size());
        const auto inSz = static_cast<Eigen::Index>(_inputSize);
        const auto ouSz = static_cast<Eigen::Index>(_layers.back().a.size());

        Eigen::MatrixXd X(inSz, B);
        Eigen::MatrixXd T(ouSz, B);
        for (Eigen::Index j = 0; j < B; ++j) {
            X.col(j) = Eigen::Map<const Eigen::VectorXd>(inputs[j].data(), inSz);
            T.col(j) = Eigen::Map<const Eigen::VectorXd>(targets[j].data(), ouSz);
        }

        // Forward: Z[l] = W[l] * A[l-1] + b[l] (broadcast)  [out_l × B]
        std::vector<Eigen::MatrixXd> A(_layers.size());
        {
            const Eigen::MatrixXd* prev = &X;
            for (size_t l = 0; l < _layers.size(); ++l) {
                Eigen::MatrixXd Z = _layers[l].W * (*prev);
                Z.colwise() += _layers[l].b;
                A[l] = Z.unaryExpr([a = _layers[l].act](double x) { return act::forward(a, x); });
                prev = &A[l];
            }
        }

        // Backward (standard batch order: all deltas use original weights).
        std::vector<Eigen::MatrixXd> D(_layers.size());
        {
            const size_t L = _layers.size() - 1;
            if (_cf == CostFunction::CrossEntropy) {
                D[L] = T - A[L];
            } else {
                Eigen::MatrixXd actD = A[L].unaryExpr(
                    [a = _layers[L].act](double y) { return act::backward(a, y); });
                D[L] = actD.cwiseProduct(T - A[L]);
            }
        }
        for (int l = static_cast<int>(_layers.size()) - 2; l >= 0; --l) {
            const size_t lu = static_cast<size_t>(l);
            Eigen::MatrixXd prop = _layers[lu + 1].W.transpose() * D[lu + 1];
            Eigen::MatrixXd actD
                = A[lu].unaryExpr([a = _layers[lu].act](double y) { return act::backward(a, y); });
            D[lu] = prop.cwiseProduct(actD);
        }

        // Weight update: mean gradient over batch.
        const double invB = 1.0 / static_cast<double>(B);
        if (_optimizer == Optimizer::Adam) {
            ++_adamT;
            const double bc1 = 1.0 - std::pow(_beta1, static_cast<double>(_adamT));
            const double bc2 = 1.0 - std::pow(_beta2, static_cast<double>(_adamT));
            for (size_t l = 0; l < _layers.size(); ++l) {
                const Eigen::MatrixXd& prevA = (l == 0) ? X : A[l - 1];
                auto& lay = _layers[l];
                const Eigen::MatrixXd gW = invB * D[l] * prevA.transpose();
                const Eigen::VectorXd gb = invB * D[l].rowwise().sum();
                lay.mW = _beta1 * lay.mW + (1.0 - _beta1) * gW;
                lay.vW = _beta2 * lay.vW + (1.0 - _beta2) * gW.cwiseProduct(gW);
                lay.mb = _beta1 * lay.mb + (1.0 - _beta1) * gb;
                lay.vb = _beta2 * lay.vb + (1.0 - _beta2) * gb.cwiseProduct(gb);
                const Eigen::MatrixXd mWhat = lay.mW / bc1;
                const Eigen::MatrixXd vWhat = lay.vW / bc2;
                const Eigen::VectorXd mbhat = lay.mb / bc1;
                const Eigen::VectorXd vbhat = lay.vb / bc2;
                lay.W += _lr
                    * mWhat.cwiseQuotient(vWhat.cwiseSqrt()
                        + Eigen::MatrixXd::Constant(vWhat.rows(), vWhat.cols(), _adamEps));
                lay.b += _lr
                    * mbhat.cwiseQuotient(
                        vbhat.cwiseSqrt() + Eigen::VectorXd::Constant(vbhat.size(), _adamEps));
            }
        } else {
            const double lrB = _lr * invB;
            for (size_t l = 0; l < _layers.size(); ++l) {
                const Eigen::MatrixXd& prevA = (l == 0) ? X : A[l - 1];
                _layers[l].dW = lrB * D[l] * prevA.transpose() + _momentum * _layers[l].dW;
                _layers[l].db = lrB * D[l].rowwise().sum() + _momentum * _layers[l].db;
                _layers[l].W += _layers[l].dW;
                _layers[l].b += _layers[l].db;
            }
        }
        return;
    }

#ifdef NUNN_HAS_ARRAYFIRE
    {
        const dim_t B = static_cast<dim_t>(inputs.size());
        const dim_t inSz = static_cast<dim_t>(_inputSize);
        const dim_t ouSz = static_cast<dim_t>(_layers.back().a.size());

        // Pack inputs and targets into column-major Eigen matrices, then upload.
        // Eigen MatrixXd is column-major, so .data() is contiguous and AF-compatible.
        Eigen::MatrixXd X_host(inSz, B);
        Eigen::MatrixXd T_host(ouSz, B);
        for (dim_t j = 0; j < B; ++j) {
            X_host.col(j) = Eigen::Map<const Eigen::VectorXd>(inputs[j].data(), inSz);
            T_host.col(j) = Eigen::Map<const Eigen::VectorXd>(targets[j].data(), ouSz);
        }
        af::array X_af = af::array(inSz, B, X_host.data(), afHost);
        af::array T_af = af::array(ouSz, B, T_host.data(), afHost);

        // Forward pass: Z[l] = W[l] * A[l-1] + tile(b[l], 1, B)  [out_l × B]
        std::vector<af::array> A_af(_layers.size());
        {
            const af::array* prev = &X_af;
            for (size_t l = 0; l < _layers.size(); ++l) {
                const af::array Z = af::matmul(*_layers[l].W_af, *prev)
                    + af::tile(*_layers[l].b_af, 1, static_cast<unsigned>(B));
                A_af[l] = af_activate(_layers[l].act, Z);
                prev = &A_af[l];
            }
        }

        // Backward (standard batch order: all deltas use original W).
        std::vector<af::array> D_af(_layers.size());
        {
            const size_t L = _layers.size() - 1;
            if (_cf == CostFunction::CrossEntropy)
                D_af[L] = T_af - A_af[L];
            else
                D_af[L] = af_activate_backward(_layers[L].act, A_af[L]) * (T_af - A_af[L]);
        }
        for (int l = static_cast<int>(_layers.size()) - 2; l >= 0; --l) {
            const size_t lu = static_cast<size_t>(l);
            const af::array prop
                = af::matmul(*_layers[lu + 1].W_af, D_af[lu + 1], AF_MAT_TRANS, AF_MAT_NONE);
            D_af[lu] = prop * af_activate_backward(_layers[lu].act, A_af[lu]);
        }

        // Weight update: mean gradient over batch.
        const double invB = 1.0 / static_cast<double>(B);
        if (_optimizer == Optimizer::Adam) {
            ++_adamT;
            const double bc1 = 1.0 - std::pow(_beta1, static_cast<double>(_adamT));
            const double bc2 = 1.0 - std::pow(_beta2, static_cast<double>(_adamT));
            for (size_t l = 0; l < _layers.size(); ++l) {
                const af::array& prevA = (l == 0) ? X_af : A_af[l - 1];
                auto& lay = _layers[l];
                const af::array gW = invB * af::matmul(D_af[l], prevA, AF_MAT_NONE, AF_MAT_TRANS);
                const af::array gb = invB * af::sum(D_af[l], 1);

                if (!lay.mW_af) {
                    lay.mW_af = af::constant(0.0, lay.W.rows(), lay.W.cols(), f64);
                    lay.vW_af = af::constant(0.0, lay.W.rows(), lay.W.cols(), f64);
                    lay.mb_af = af::constant(0.0, static_cast<dim_t>(lay.b.size()), (dim_t)1, f64);
                    lay.vb_af = af::constant(0.0, static_cast<dim_t>(lay.b.size()), (dim_t)1, f64);
                }

                *lay.mW_af = _beta1 * *lay.mW_af + (1.0 - _beta1) * gW;
                *lay.vW_af = _beta2 * *lay.vW_af + (1.0 - _beta2) * gW * gW;
                *lay.mb_af = _beta1 * *lay.mb_af + (1.0 - _beta1) * gb;
                *lay.vb_af = _beta2 * *lay.vb_af + (1.0 - _beta2) * gb * gb;

                const af::array mWhat = *lay.mW_af / bc1;
                const af::array vWhat = *lay.vW_af / bc2;
                const af::array mbhat = *lay.mb_af / bc1;
                const af::array vbhat = *lay.vb_af / bc2;
                *lay.W_af += _lr * mWhat / (af::sqrt(vWhat) + _adamEps);
                *lay.b_af += _lr * mbhat / (af::sqrt(vbhat) + _adamEps);
            }
        } else {
            const double lrB = _lr * invB;
            for (size_t l = 0; l < _layers.size(); ++l) {
                const af::array& prevA = (l == 0) ? X_af : A_af[l - 1];
                _layers[l].dW_af = lrB * af::matmul(D_af[l], prevA, AF_MAT_NONE, AF_MAT_TRANS)
                    + _momentum * *_layers[l].dW_af;
                _layers[l].db_af = lrB * af::sum(D_af[l], 1) + _momentum * *_layers[l].db_af;
                *_layers[l].W_af += *_layers[l].dW_af;
                *_layers[l].b_af += *_layers[l].db_af;
            }
        }

        // Sync last sample's output to host for metrics called after trainBatch.
        A_af.back()(af::span, B - 1).host(_layers.back().a.data());
    }
#endif
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

// ── Layer inspection ──────────────────────────────────────────────────────────

const Eigen::VectorXd& MlpMatrixNN::getLayerOutput(size_t layer) const
{
    const auto& l = _layers.at(layer);
#ifdef NUNN_HAS_ARRAYFIRE
    if (_backend == ComputeBackend::OpenCL && l.a_af)
        l.a_af->host(l.a.data());
#endif
    return l.a;
}

Eigen::MatrixXd MlpMatrixNN::getLayerW(size_t layer) const
{
    const auto& l = _layers.at(layer);
    Eigen::MatrixXd W = l.W;
#ifdef NUNN_HAS_ARRAYFIRE
    if (_backend == ComputeBackend::OpenCL && l.W_af)
        l.W_af->host(W.data());
#endif
    return W;
}

Eigen::VectorXd MlpMatrixNN::getLayerB(size_t layer) const
{
    const auto& l = _layers.at(layer);
    Eigen::VectorXd b = l.b;
#ifdef NUNN_HAS_ARRAYFIRE
    if (_backend == ComputeBackend::OpenCL && l.b_af)
        l.b_af->host(b.data());
#endif
    return b;
}

void MlpMatrixNN::setLayerW(size_t layer, const Eigen::MatrixXd& W)
{
    auto& l = _layers.at(layer);
    l.W = W;

#ifdef NUNN_HAS_ARRAYFIRE
    if (_backend == ComputeBackend::OpenCL)
        l.W_af = af::array(l.W.rows(), l.W.cols(), l.W.data(), afHost);
#endif
}

void MlpMatrixNN::setLayerB(size_t layer, const Eigen::VectorXd& b)
{
    auto& l = _layers.at(layer);
    l.b = b;

#ifdef NUNN_HAS_ARRAYFIRE
    if (_backend == ComputeBackend::OpenCL)
        l.b_af = af::array(static_cast<dim_t>(l.b.size()), (dim_t)1, l.b.data(), afHost);
#endif
}

// ── getInputGradient ──────────────────────────────────────────────────────────

Eigen::VectorXd MlpMatrixNN::getInputGradient() const
{
    // dL/d_input = W[0]^T * delta[0], valid after backPropagate() (Eigen path).
    return _layers[0].W.transpose() * _layers[0].delta;
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

// ── Topology ──────────────────────────────────────────────────────────────────

std::vector<size_t> MlpMatrixNN::getTopology() const
{
    std::vector<size_t> topo;
    topo.reserve(_layers.size() + 1);
    topo.push_back(_inputSize);
    for (const auto& l : _layers)
        topo.push_back(static_cast<size_t>(l.W.rows()));
    return topo;
}

// ── JSON persistence ──────────────────────────────────────────────────────────

namespace {
    static std::string_view cfName(CostFunction cf) noexcept
    {
        return cf == CostFunction::CrossEntropy ? "cross_entropy" : "mse";
    }
    static CostFunction cfFromString(std::string_view s) noexcept
    {
        return s == "cross_entropy" ? CostFunction::CrossEntropy : CostFunction::MSE;
    }
    static std::string_view optName(MlpMatrixNN::Optimizer opt) noexcept
    {
        return opt == MlpMatrixNN::Optimizer::Adam ? "adam" : "sgd";
    }
    static MlpMatrixNN::Optimizer optFromString(std::string_view s) noexcept
    {
        return s == "adam" ? MlpMatrixNN::Optimizer::Adam : MlpMatrixNN::Optimizer::SGD;
    }
} // namespace

std::ostream& MlpMatrixNN::toJson(std::ostream& os) const
{
    using json = nlohmann::json;

    json j;
    j["model_type"] = "mlp_matrix";
    j["version"] = 1;
    j["learningRate"] = _lr;
    j["momentum"] = _momentum;
    j["optimizer"] = std::string(optName(_optimizer));
    j["costFunction"] = std::string(cfName(_cf));

    json jlayers = json::array();
    // First entry: input descriptor (size only, no weights)
    {
        json entry;
        entry["size"] = _inputSize;
        entry["activation"] = "none";
        jlayers.push_back(std::move(entry));
    }
    for (const auto& l : _layers) {
        Eigen::MatrixXd W = l.W;
        Eigen::VectorXd b = l.b;
#ifdef NUNN_HAS_ARRAYFIRE
        if (_backend == ComputeBackend::OpenCL) {
            if (l.W_af)
                l.W_af->host(W.data());
            if (l.b_af)
                l.b_af->host(b.data());
        }
#endif

        json entry;
        entry["size"] = static_cast<size_t>(W.rows());
        entry["activation"] = std::string(act::name(l.act));
        // W as array of rows
        json jW = json::array();
        for (Eigen::Index r = 0; r < W.rows(); ++r) {
            json row = json::array();
            for (Eigen::Index c = 0; c < W.cols(); ++c)
                row.push_back(W(r, c));
            jW.push_back(std::move(row));
        }
        entry["W"] = std::move(jW);
        // b as flat array
        json jb = json::array();
        for (Eigen::Index i = 0; i < b.size(); ++i)
            jb.push_back(b(i));
        entry["b"] = std::move(jb);
        jlayers.push_back(std::move(entry));
    }
    j["layers"] = std::move(jlayers);

    os << j.dump(2);
    return os;
}

std::istream& MlpMatrixNN::loadJson(std::istream& is)
{
    using json = nlohmann::json;

    const json j = json::parse(is);

    if (j.value("model_type", "") != "mlp_matrix")
        throw std::runtime_error("MlpMatrixNN::loadJson: unexpected model_type");

    const double lr = j.at("learningRate").get<double>();
    const double momentum = j.at("momentum").get<double>();
    const auto cf = cfFromString(j.value("costFunction", "mse"));
    const auto opt = optFromString(j.value("optimizer", "sgd"));

    const auto& jlayers = j.at("layers");
    if (jlayers.size() < 2)
        throw std::runtime_error("MlpMatrixNN::loadJson: need at least input + one neuron layer");

    // Reconstruct LayerConfig from JSON
    std::vector<LayerConfig> configs;
    configs.reserve(jlayers.size());
    for (size_t li = 0; li < jlayers.size(); ++li) {
        const auto& entry = jlayers[li];
        const size_t sz = entry.at("size").get<size_t>();
        const std::string actStr = entry.value("activation", "sigmoid");
        const Activation act = (actStr == "none") ? Activation::Sigmoid : act::fromString(actStr);
        configs.emplace_back(sz, act);
    }

    // Reconstruct the network (this replaces *this)
    *this = MlpMatrixNN(configs, lr, momentum, cf);
    setOptimizer(opt);

    // Restore weights
    for (size_t li = 1; li < jlayers.size(); ++li) {
        const auto& entry = jlayers[li];
        const auto& jW = entry.at("W");
        const auto& jb = entry.at("b");

        const size_t nrows = jW.size();
        const size_t ncols = jW.at(0).size();
        Eigen::MatrixXd W(static_cast<Eigen::Index>(nrows), static_cast<Eigen::Index>(ncols));
        for (size_t r = 0; r < nrows; ++r)
            for (size_t c = 0; c < ncols; ++c)
                W(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c))
                    = jW[r][c].get<double>();
        setLayerW(li - 1, W);

        Eigen::VectorXd b(static_cast<Eigen::Index>(jb.size()));
        for (size_t i = 0; i < jb.size(); ++i)
            b(static_cast<Eigen::Index>(i)) = jb[i].get<double>();
        setLayerB(li - 1, b);
    }

    return is;
}

} // namespace nu
