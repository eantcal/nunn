//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_transformer.h"

#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nu {

// ── Internal helpers ──────────────────────────────────────────────────────────

// Row-wise softmax (numerically stable).
static Eigen::MatrixXd rowSoftmax(const Eigen::MatrixXd& x)
{
    Eigen::MatrixXd out(x.rows(), x.cols());
    for (Eigen::Index r = 0; r < x.rows(); ++r) {
        const double mx = x.row(r).maxCoeff();
        out.row(r) = (x.row(r).array() - mx).exp();
        out.row(r) /= out.row(r).sum();
    }
    return out;
}

// Jacobian-vector product for softmax: given softmax output `a` and upstream
// grad `g`, returns grad w.r.t. the pre-softmax scores.
// d_scores[r] = a[r] * (g[r] - dot(g[r], a[r]))
static Eigen::MatrixXd softmaxBackward(const Eigen::MatrixXd& g, const Eigen::MatrixXd& a)
{
    Eigen::MatrixXd ds(g.rows(), g.cols());
    for (Eigen::Index r = 0; r < g.rows(); ++r) {
        const double dot = g.row(r).dot(a.row(r));
        ds.row(r) = a.row(r).array() * (g.row(r).array() - dot);
    }
    return ds;
}

// Xavier (Glorot) normal initialisation.
static Eigen::MatrixXd xavierInit(Eigen::Index rows, Eigen::Index cols, std::mt19937& rng)
{
    std::normal_distribution<double> d(0.0, std::sqrt(1.0 / static_cast<double>(cols)));
    Eigen::MatrixXd W(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r)
        for (Eigen::Index c = 0; c < cols; ++c)
            W(r, c) = d(rng);
    return W;
}

// He normal initialisation (for ReLU layers).
static Eigen::MatrixXd heInit(Eigen::Index rows, Eigen::Index cols, std::mt19937& rng)
{
    std::normal_distribution<double> d(0.0, std::sqrt(2.0 / static_cast<double>(cols)));
    Eigen::MatrixXd W(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r)
        for (Eigen::Index c = 0; c < cols; ++c)
            W(r, c) = d(rng);
    return W;
}

// ── LayerNorm ─────────────────────────────────────────────────────────────────

LayerNorm::LayerNorm(size_t dModel, double eps)
    : _d(dModel)
    , _eps(eps)
    , _gamma(Eigen::VectorXd::Ones(static_cast<Eigen::Index>(dModel)))
    , _beta(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dModel)))
{
}

Eigen::MatrixXd LayerNorm::forward(const Eigen::MatrixXd& x)
{
    _xin = x;
    const Eigen::Index T = x.rows(), d = x.cols();
    _xhat.resize(T, d);
    _invStd.resize(T);
    Eigen::MatrixXd out(T, d);

    for (Eigen::Index r = 0; r < T; ++r) {
        const double mu = x.row(r).mean();
        const double var = (x.row(r).array() - mu).square().mean();
        _invStd(r) = 1.0 / std::sqrt(var + _eps);
        _xhat.row(r) = (x.row(r).array() - mu) * _invStd(r);
        out.row(r) = _xhat.row(r).array() * _gamma.array() + _beta.array();
    }
    return out;
}

Eigen::MatrixXd LayerNorm::backward(const Eigen::MatrixXd& grad, double lr)
{
    const Eigen::Index T = grad.rows();
    const double d = static_cast<double>(_d);

    // Param gradients.
    Eigen::VectorXd dGamma = (_xhat.array() * grad.array()).matrix().colwise().sum().transpose();
    Eigen::VectorXd dBeta = grad.colwise().sum().transpose();

    // Gradient w.r.t. xhat.
    Eigen::MatrixXd dXhat = grad.array().rowwise() * _gamma.array().transpose();

    // Gradient w.r.t. input (standard LN backward formula per row).
    Eigen::MatrixXd dX(T, static_cast<Eigen::Index>(_d));
    for (Eigen::Index r = 0; r < T; ++r) {
        const auto g = dXhat.row(r);
        const double mean_g = g.mean();
        const double mean_g_xhat = g.dot(_xhat.row(r)) / d;
        dX.row(r) = _invStd(r) * (g.array() - mean_g - _xhat.row(r).array() * mean_g_xhat);
    }

    _gamma -= lr * dGamma;
    _beta -= lr * dBeta;

    return dX;
}

// ── SelfAttentionLayer ────────────────────────────────────────────────────────

SelfAttentionLayer::SelfAttentionLayer(size_t dModel, size_t numHeads, double lr)
    : _d(dModel)
    , _h(numHeads)
    , _dk(dModel / numHeads)
    , _lr(lr)
    , _WO(Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(dModel), static_cast<Eigen::Index>(dModel)))
    , _bO(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dModel)))
{
    if (dModel % numHeads != 0)
        throw std::invalid_argument("SelfAttentionLayer: dModel must be divisible by numHeads");

    std::mt19937 rng(std::random_device{}());
    const Eigen::Index d = static_cast<Eigen::Index>(dModel);
    const Eigen::Index dk = static_cast<Eigen::Index>(_dk);

    for (size_t h = 0; h < numHeads; ++h) {
        _WQh.push_back(xavierInit(d, dk, rng));
        _WKh.push_back(xavierInit(d, dk, rng));
        _WVh.push_back(xavierInit(d, dk, rng));
    }
    _WO = xavierInit(d, d, rng);
}

Eigen::MatrixXd SelfAttentionLayer::forward(const Eigen::MatrixXd& x, bool causal)
{
    _xin = x;
    const Eigen::Index T = x.rows(), d = static_cast<Eigen::Index>(_d);
    const Eigen::Index dk = static_cast<Eigen::Index>(_dk);
    const double scale = 1.0 / std::sqrt(static_cast<double>(_dk));

    _Qh.resize(_h);
    _Kh.resize(_h);
    _Vh.resize(_h);
    _Attn_h.resize(_h);
    _headOut_h.resize(_h);
    _concat.resize(T, d);

    for (size_t hi = 0; hi < _h; ++hi) {
        _Qh[hi] = x * _WQh[hi]; // [T × dk]
        _Kh[hi] = x * _WKh[hi];
        _Vh[hi] = x * _WVh[hi];

        Eigen::MatrixXd scores = _Qh[hi] * _Kh[hi].transpose() * scale; // [T × T]

        if (causal)
            for (Eigen::Index r = 0; r < T; ++r)
                for (Eigen::Index c = r + 1; c < T; ++c)
                    scores(r, c) = -1e9;

        _Attn_h[hi] = rowSoftmax(scores);
        _headOut_h[hi] = _Attn_h[hi] * _Vh[hi]; // [T × dk]

        _concat.block(0, static_cast<Eigen::Index>(hi * _dk), T, dk) = _headOut_h[hi];
    }

    Eigen::MatrixXd out = _concat * _WO;
    out.rowwise() += _bO.transpose();
    return out;
}

Eigen::MatrixXd SelfAttentionLayer::backward(const Eigen::MatrixXd& gradOut, double lr)
{
    const double useLr = (lr > 0.0) ? lr : _lr;
    const Eigen::Index T = gradOut.rows(), d = static_cast<Eigen::Index>(_d);
    const Eigen::Index dk = static_cast<Eigen::Index>(_dk);
    const double scale = 1.0 / std::sqrt(static_cast<double>(_dk));

    // Backward through output projection.
    Eigen::MatrixXd dWO = _concat.transpose() * gradOut; // [d × d]
    Eigen::VectorXd dbO = gradOut.colwise().sum().transpose();
    Eigen::MatrixXd dConcat = gradOut * _WO.transpose(); // [T × d]

    Eigen::MatrixXd dX = Eigen::MatrixXd::Zero(T, d);

    for (size_t hi = 0; hi < _h; ++hi) {
        Eigen::MatrixXd dHead = dConcat.block(0, static_cast<Eigen::Index>(hi * _dk), T, dk);

        // Backward through attention output = Attn * V.
        Eigen::MatrixXd dV = _Attn_h[hi].transpose() * dHead; // [T × dk]
        Eigen::MatrixXd dAttn = dHead * _Vh[hi].transpose(); // [T × T]
        Eigen::MatrixXd dScores = softmaxBackward(dAttn, _Attn_h[hi]) * scale; // [T × T]

        // Backward through Q * K^T.
        Eigen::MatrixXd dQ = dScores * _Kh[hi]; // [T × dk]
        Eigen::MatrixXd dK = dScores.transpose() * _Qh[hi]; // [T × dk]

        // Accumulate weight grads and input grad.
        _WQh[hi] -= useLr * (_xin.transpose() * dQ);
        _WKh[hi] -= useLr * (_xin.transpose() * dK);
        _WVh[hi] -= useLr * (_xin.transpose() * dV);

        dX += dQ * _WQh[hi].transpose();
        dX += dK * _WKh[hi].transpose();
        dX += dV * _WVh[hi].transpose();
    }

    _WO -= useLr * dWO;
    _bO -= useLr * dbO;

    return dX;
}

// ── TransformerBlock ──────────────────────────────────────────────────────────

TransformerBlock::TransformerBlock(size_t dModel, size_t numHeads, size_t dFF, double lr)
    : _lr(lr)
    , _ln1(dModel)
    , _ln2(dModel)
    , _attn(dModel, numHeads, lr)
{
    std::mt19937 rng(std::random_device{}());
    const Eigen::Index d = static_cast<Eigen::Index>(dModel);
    const Eigen::Index f = static_cast<Eigen::Index>(dFF);

    _W1 = heInit(d, f, rng); // [d × dFF]  fan_in = dFF (columns)
    _b1 = Eigen::VectorXd::Zero(f);
    _W2 = heInit(f, d, rng); // [dFF × d]  fan_in = d
    _b2 = Eigen::VectorXd::Zero(d);
}

Eigen::MatrixXd TransformerBlock::forward(const Eigen::MatrixXd& x, bool causal)
{
    _xin = x;

    // Pre-LN attention sublayer.
    _ln1out = _ln1.forward(x);
    _attnOut = _attn.forward(_ln1out, causal);
    Eigen::MatrixXd r1 = x + _attnOut; // residual

    // Pre-LN FFN sublayer.
    _ln2out = _ln2.forward(r1);
    Eigen::MatrixXd h = _ln2out * _W1; // [T × dFF]
    h.rowwise() += _b1.transpose();
    _h1act = h.cwiseMax(0.0); // ReLU
    Eigen::MatrixXd ff = _h1act * _W2;
    ff.rowwise() += _b2.transpose();

    return r1 + ff; // residual
}

Eigen::MatrixXd TransformerBlock::backward(const Eigen::MatrixXd& gradOut, double lr)
{
    const double useLr = (lr > 0.0) ? lr : _lr;
    const Eigen::Index T = gradOut.rows();

    // Residual: grad flows to both r1 and ff branches.
    Eigen::MatrixXd dR1 = gradOut;
    Eigen::MatrixXd dFF = gradOut;

    // Backward through FFN.
    Eigen::MatrixXd dW2 = _h1act.transpose() * dFF; // [dFF × d]
    Eigen::VectorXd db2 = dFF.colwise().sum().transpose();
    Eigen::MatrixXd dH1act = dFF * _W2.transpose(); // [T × dFF]

    // Backward through ReLU using saved post-activation (_h1act > 0).
    Eigen::MatrixXd dH1 = dH1act.array() * (_h1act.array() > 0.0).cast<double>();

    Eigen::MatrixXd dW1 = _ln2out.transpose() * dH1; // [d × dFF]
    Eigen::VectorXd db1 = dH1.colwise().sum().transpose();
    Eigen::MatrixXd dN2 = dH1 * _W1.transpose(); // [T × d]

    _W2 -= useLr * dW2;
    _b2 -= useLr * db2;
    _W1 -= useLr * dW1;
    _b1 -= useLr * db1;

    // Backward through LN2 (pre-norm of FFN sublayer).
    dR1 += _ln2.backward(dN2, useLr);

    // Residual: grad from r1 = x + attnOut splits to x and attnOut paths.
    Eigen::MatrixXd dAttnOut = dR1;
    Eigen::MatrixXd dX = dR1; // residual connection to input

    // Backward through attention.
    Eigen::MatrixXd dN1 = _attn.backward(dAttnOut, useLr); // [T × d]

    // Backward through LN1 (pre-norm of attention sublayer).
    dX += _ln1.backward(dN1, useLr);

    return dX;
}

// ── MiniTransformer ───────────────────────────────────────────────────────────

Eigen::MatrixXd MiniTransformer::_makePosEnc(size_t T, size_t d)
{
    Eigen::MatrixXd pe(static_cast<Eigen::Index>(T), static_cast<Eigen::Index>(d));
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < d; i += 2) {
            const double freq = 1.0 / std::pow(10000.0, static_cast<double>(i) / d);
            pe(static_cast<Eigen::Index>(t), static_cast<Eigen::Index>(i))
                = std::sin(static_cast<double>(t) * freq);
            if (i + 1 < d)
                pe(static_cast<Eigen::Index>(t), static_cast<Eigen::Index>(i + 1))
                    = std::cos(static_cast<double>(t) * freq);
        }
    }
    return pe;
}

MiniTransformer::MiniTransformer(size_t vocabSize, size_t seqLen, size_t dModel, size_t numHeads,
    size_t dFF, size_t numLayers, double lr)
    : _V(vocabSize)
    , _T(seqLen)
    , _d(dModel)
    , _lr(lr)
    , _posEnc(_makePosEnc(seqLen, dModel))
    , _Wout(Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(dModel), static_cast<Eigen::Index>(vocabSize)))
    , _bout(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(vocabSize)))
{
    std::mt19937 rng(std::random_device{}());
    const Eigen::Index V = static_cast<Eigen::Index>(vocabSize);
    const Eigen::Index d = static_cast<Eigen::Index>(dModel);

    _embed = xavierInit(V, d, rng);
    _Wout = xavierInit(d, V, rng);

    for (size_t l = 0; l < numLayers; ++l)
        _blocks.emplace_back(dModel, numHeads, dFF, lr);
}

Eigen::MatrixXd MiniTransformer::forward(const std::vector<int>& tokens)
{
    assert(tokens.size() == _T);
    _lastTokens = tokens;

    const Eigen::Index T = static_cast<Eigen::Index>(_T);
    const Eigen::Index d = static_cast<Eigen::Index>(_d);

    // Build embedded + positional-encoded input.
    _xemb.resize(T, d);
    for (Eigen::Index t = 0; t < T; ++t)
        _xemb.row(t) = _embed.row(tokens[static_cast<size_t>(t)]) + _posEnc.row(t);

    // Forward through transformer blocks (causal mask for LM).
    Eigen::MatrixXd x = _xemb;
    for (auto& block : _blocks)
        x = block.forward(x, /*causal=*/true);
    _xfinal = x;

    // Output projection: [T × V].
    Eigen::MatrixXd logits = x * _Wout;
    logits.rowwise() += _bout.transpose();
    return logits;
}

double MiniTransformer::train(const std::vector<int>& inputs, const std::vector<int>& targets)
{
    assert(inputs.size() == _T && targets.size() == _T);

    const Eigen::MatrixXd logits = forward(inputs);
    const Eigen::MatrixXd probs = rowSoftmax(logits);

    const Eigen::Index T = static_cast<Eigen::Index>(_T);
    const Eigen::Index V = static_cast<Eigen::Index>(_V);

    // Cross-entropy loss and its gradient w.r.t. logits.
    double loss = 0.0;
    Eigen::MatrixXd dLogits = probs; // copy; we'll subtract 1 at target positions
    for (Eigen::Index t = 0; t < T; ++t) {
        const int tgt = targets[static_cast<size_t>(t)];
        loss -= std::log(std::max(probs(t, tgt), 1e-9));
        dLogits(t, tgt) -= 1.0;
    }
    dLogits /= static_cast<double>(_T); // mean over sequence

    // Backward through output projection.
    Eigen::MatrixXd dWout = _xfinal.transpose() * dLogits; // [d × V]
    Eigen::VectorXd dbout = dLogits.colwise().sum().transpose();
    Eigen::MatrixXd dX = dLogits * _Wout.transpose(); // [T × d]

    _Wout -= _lr * dWout;
    _bout -= _lr * dbout;

    // Backward through transformer blocks (reverse order).
    for (int i = static_cast<int>(_blocks.size()) - 1; i >= 0; --i)
        dX = _blocks[static_cast<size_t>(i)].backward(dX, 0.0);

    // Backward through embedding (positional encoding is fixed).
    for (Eigen::Index t = 0; t < T; ++t)
        _embed.row(_lastTokens[static_cast<size_t>(t)]) -= _lr * dX.row(t);

    return loss / static_cast<double>(_T);
}

std::vector<int> MiniTransformer::generate(
    const std::vector<int>& prompt, size_t nTokens, double temperature, std::mt19937* rng)
{
    std::vector<int> context(prompt);
    std::vector<int> generated;

    std::mt19937 localRng(42);
    std::mt19937& gen = rng ? *rng : localRng;

    for (size_t i = 0; i < nTokens; ++i) {
        // Build a seqLen-sized context window.
        std::vector<int> window;
        if (context.size() >= _T) {
            window.assign(context.end() - static_cast<ptrdiff_t>(_T), context.end());
        } else {
            // Pad with zeros at the front.
            window.assign(_T - context.size(), 0);
            window.insert(window.end(), context.begin(), context.end());
        }

        // Forward and sample from the last position.
        const Eigen::MatrixXd logits = forward(window);
        const Eigen::Index lastRow = static_cast<Eigen::Index>(_T) - 1;
        Eigen::VectorXd scaled = logits.row(lastRow).transpose() / temperature;
        const double mx = scaled.maxCoeff();
        scaled = (scaled.array() - mx).exp();
        scaled /= scaled.sum();

        std::discrete_distribution<int> dist(scaled.data(), scaled.data() + scaled.size());
        const int next = dist(gen);
        context.push_back(next);
        generated.push_back(next);
    }
    return generated;
}

} // namespace nu
