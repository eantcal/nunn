//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// 1D convolutional and max-pooling layers for building ConvNet pipelines.
// Vectors use channel-major flat layout: all values of channel 0 first,
// then channel 1, etc.
//

#pragma once

#include "nu_activation.h"

#include <Eigen/Core>
#include <memory>
#include <stdexcept>
#include <vector>

namespace nu {

// ── Abstract base layer ───────────────────────────────────────────────────────

class IConvLayer1D {
public:
    virtual ~IConvLayer1D() = default;

    // Forward pass; returns reference to internal output buffer.
    virtual const std::vector<double>& forward(const std::vector<double>& in) = 0;

    // Backward pass; lr is used only by layers that have trainable parameters.
    // Returns the gradient w.r.t. the layer input (grad_in).
    virtual std::vector<double> backward(const std::vector<double>& gradOut, double lr) = 0;

    virtual size_t outChannels() const noexcept = 0;
    virtual size_t outLength() const noexcept = 0;
    size_t outputSize() const noexcept { return outChannels() * outLength(); }
};

// ── Conv1DLayer ───────────────────────────────────────────────────────────────
//
// 1D convolution with valid padding and stride 1.
//   input  layout: [inChannels × inLength]   (flat, channel-major)
//   output layout: [outChannels × outLength]
//   outLength = inLength - kernelSize + 1
//
// Activation applied element-wise after the convolution.

class Conv1DLayer : public IConvLayer1D {
public:
    Conv1DLayer(size_t inChannels, size_t inLength, size_t outChannels, size_t kernelSize,
        Activation act = Activation::ReLU, double lr = 0.01);

    const std::vector<double>& forward(const std::vector<double>& in) override;
    std::vector<double> backward(const std::vector<double>& gradOut, double lr) override;

    size_t outChannels() const noexcept override { return _outCh; }
    size_t outLength() const noexcept override { return _outLen; }

    void reshuffleWeights();

private:
    size_t _inCh, _inLen, _outCh, _K, _outLen;
    Activation _act;
    double _lr;

    Eigen::MatrixXd _W; // [outCh × inCh*K]
    Eigen::VectorXd _b; // [outCh]
    Eigen::MatrixXd _Xcol; // [inCh*K × outLen] — saved in forward for backward
    Eigen::MatrixXd _Yact; // [outCh × outLen]  — activation output, for backward

    std::vector<double> _out; // flat output [outCh * outLen]
};

// ── MaxPool1DLayer ────────────────────────────────────────────────────────────
//
// Non-overlapping max pooling.
//   input  layout: [channels × inLength]   (flat, channel-major)
//   output layout: [channels × outLength]
//   outLength = inLength / poolSize  (integer floor; remainder is discarded)

class MaxPool1DLayer : public IConvLayer1D {
public:
    MaxPool1DLayer(size_t channels, size_t inLength, size_t poolSize);

    const std::vector<double>& forward(const std::vector<double>& in) override;
    std::vector<double> backward(const std::vector<double>& gradOut, double lr) override;

    size_t outChannels() const noexcept override { return _ch; }
    size_t outLength() const noexcept override { return _outLen; }

private:
    size_t _ch, _inLen, _P, _outLen;
    std::vector<size_t> _maxIdx; // index of selected max per output cell
    std::vector<double> _out; // flat output [ch * outLen]
};

} // namespace nu
