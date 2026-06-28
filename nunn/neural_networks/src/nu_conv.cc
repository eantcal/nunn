//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_conv.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <random>

namespace nu {

// ── Conv1DLayer ───────────────────────────────────────────────────────────────

Conv1DLayer::Conv1DLayer(size_t inChannels, size_t inLength, size_t outChannels, size_t kernelSize,
    Activation act, double lr)
    : _inCh(inChannels)
    , _inLen(inLength)
    , _outCh(outChannels)
    , _K(kernelSize)
    , _outLen(inLength >= kernelSize ? inLength - kernelSize + 1 : 0)
    , _act(act)
    , _lr(lr)
    , _W(Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(outChannels),
          static_cast<Eigen::Index>(inChannels * kernelSize)))
    , _b(Eigen::VectorXd::Zero(static_cast<Eigen::Index>(outChannels)))
    , _Xcol(Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(inChannels * kernelSize), static_cast<Eigen::Index>(_outLen)))
    , _Yact(Eigen::MatrixXd::Zero(
          static_cast<Eigen::Index>(outChannels), static_cast<Eigen::Index>(_outLen)))
    , _out(outChannels * _outLen, 0.0)
{
    if (inChannels == 0 || outChannels == 0 || kernelSize == 0)
        throw std::invalid_argument("Conv1DLayer: dimensions must be > 0");
    if (inLength < kernelSize)
        throw std::invalid_argument("Conv1DLayer: inLength must be >= kernelSize");
    reshuffleWeights();
}

void Conv1DLayer::reshuffleWeights()
{
    std::mt19937 rng(std::random_device{}());
    // He initialization for ReLU/LeakyReLU, Xavier otherwise.
    const double fan_in = static_cast<double>(_inCh * _K);
    const double scale = (_act == Activation::ReLU || _act == Activation::LeakyReLU)
        ? std::sqrt(2.0 / fan_in)
        : std::sqrt(1.0 / fan_in);
    std::normal_distribution<double> dist(0.0, scale);
    for (Eigen::Index r = 0; r < _W.rows(); ++r)
        for (Eigen::Index c = 0; c < _W.cols(); ++c)
            _W(r, c) = dist(rng);
    _b.setZero();
}

const std::vector<double>& Conv1DLayer::forward(const std::vector<double>& in)
{
    assert(in.size() == _inCh * _inLen);

    // Build im2col matrix: _Xcol [inCh*K × outLen]
    for (size_t t = 0; t < _outLen; ++t)
        for (size_t ci = 0; ci < _inCh; ++ci)
            for (size_t k = 0; k < _K; ++k)
                _Xcol(static_cast<Eigen::Index>(ci * _K + k), static_cast<Eigen::Index>(t))
                    = in[ci * _inLen + t + k];

    // Y_pre = W * Xcol + b (broadcast)  [outCh × outLen]
    Eigen::MatrixXd Ypre = _W * _Xcol;
    Ypre.colwise() += _b;

    // Apply activation; save _Yact for backward.
    _Yact = Ypre.unaryExpr([a = _act](double x) { return act::forward(a, x); });

    // Flatten to output vector.
    for (size_t co = 0; co < _outCh; ++co)
        for (size_t t = 0; t < _outLen; ++t)
            _out[co * _outLen + t]
                = _Yact(static_cast<Eigen::Index>(co), static_cast<Eigen::Index>(t));

    return _out;
}

std::vector<double> Conv1DLayer::backward(const std::vector<double>& gradOut, double lr)
{
    assert(gradOut.size() == _outCh * _outLen);

    // Unflatten incoming gradient.
    Eigen::MatrixXd dYact(static_cast<Eigen::Index>(_outCh), static_cast<Eigen::Index>(_outLen));
    for (size_t co = 0; co < _outCh; ++co)
        for (size_t t = 0; t < _outLen; ++t)
            dYact(static_cast<Eigen::Index>(co), static_cast<Eigen::Index>(t))
                = gradOut[co * _outLen + t];

    // Activation backward: chain rule through activation.
    const Eigen::MatrixXd dY
        = dYact.cwiseProduct(_Yact.unaryExpr([a = _act](double y) { return act::backward(a, y); }));

    // Weight gradients.
    const Eigen::MatrixXd dW = dY * _Xcol.transpose(); // [outCh × inCh*K]
    const Eigen::VectorXd db = dY.rowwise().sum(); // [outCh]

    // Input gradient (before update, while W is still unchanged).
    const Eigen::MatrixXd dXcol = _W.transpose() * dY; // [inCh*K × outLen]

    // SGD update.
    const double useLr = (lr > 0.0) ? lr : _lr;
    _W -= useLr * dW;
    _b -= useLr * db;

    // Col2im: accumulate dXcol → grad_in [inCh * inLen]
    std::vector<double> gradIn(_inCh * _inLen, 0.0);
    for (size_t t = 0; t < _outLen; ++t)
        for (size_t ci = 0; ci < _inCh; ++ci)
            for (size_t k = 0; k < _K; ++k)
                gradIn[ci * _inLen + t + k]
                    += dXcol(static_cast<Eigen::Index>(ci * _K + k), static_cast<Eigen::Index>(t));

    return gradIn;
}

// ── MaxPool1DLayer ────────────────────────────────────────────────────────────

MaxPool1DLayer::MaxPool1DLayer(size_t channels, size_t inLength, size_t poolSize)
    : _ch(channels)
    , _inLen(inLength)
    , _P(poolSize)
    , _outLen(poolSize > 0 ? inLength / poolSize : 0)
    , _maxIdx(channels * (poolSize > 0 ? inLength / poolSize : 0), 0)
    , _out(channels * (poolSize > 0 ? inLength / poolSize : 0), 0.0)
{
    if (channels == 0 || inLength == 0 || poolSize == 0)
        throw std::invalid_argument("MaxPool1DLayer: dimensions must be > 0");
    if (inLength < poolSize)
        throw std::invalid_argument("MaxPool1DLayer: inLength must be >= poolSize");
}

const std::vector<double>& MaxPool1DLayer::forward(const std::vector<double>& in)
{
    assert(in.size() == _ch * _inLen);

    for (size_t c = 0; c < _ch; ++c) {
        for (size_t w = 0; w < _outLen; ++w) {
            double maxVal = -std::numeric_limits<double>::max();
            size_t maxPos = c * _inLen + w * _P;
            for (size_t p = 0; p < _P; ++p) {
                const size_t pos = c * _inLen + w * _P + p;
                if (in[pos] > maxVal) {
                    maxVal = in[pos];
                    maxPos = pos;
                }
            }
            _out[c * _outLen + w] = maxVal;
            _maxIdx[c * _outLen + w] = maxPos;
        }
    }
    return _out;
}

std::vector<double> MaxPool1DLayer::backward(const std::vector<double>& gradOut, double /*lr*/)
{
    assert(gradOut.size() == _ch * _outLen);
    std::vector<double> gradIn(_ch * _inLen, 0.0);
    for (size_t c = 0; c < _ch; ++c)
        for (size_t w = 0; w < _outLen; ++w)
            gradIn[_maxIdx[c * _outLen + w]] += gradOut[c * _outLen + w];
    return gradIn;
}

} // namespace nu
