//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_convnet.h"

#include <stdexcept>

namespace nu {

// ── Construction ──────────────────────────────────────────────────────────────

ConvNet::ConvNet(size_t inChannels, size_t inLength)
    : _inCh(inChannels)
    , _inLen(inLength)
    , _curCh(inChannels)
    , _curLen(inLength)
{
    if (inChannels == 0 || inLength == 0)
        throw std::invalid_argument("ConvNet: inChannels and inLength must be > 0");
}

// ── Layer builders ────────────────────────────────────────────────────────────

ConvNet& ConvNet::addConv1D(size_t outChannels, size_t kernelSize, Activation act, double lr)
{
    _layers.push_back(
        std::make_unique<Conv1DLayer>(_curCh, _curLen, outChannels, kernelSize, act, lr));
    _curCh = outChannels;
    _curLen = _curLen - kernelSize + 1;
    return *this;
}

ConvNet& ConvNet::addMaxPool1D(size_t poolSize)
{
    _layers.push_back(std::make_unique<MaxPool1DLayer>(_curCh, _curLen, poolSize));
    _curLen = _curLen / poolSize;
    return *this;
}

void ConvNet::setFCHead(const std::vector<MlpMatrixNN::LayerConfig>& fcLayers, double lr)
{
    if (fcLayers.empty() || fcLayers[0].size != flatFeatureSize())
        throw std::invalid_argument(
            "ConvNet::setFCHead: fcLayers[0].size must match flatFeatureSize()");
    _fc = std::make_unique<MlpMatrixNN>(fcLayers, lr);
}

// ── Properties ────────────────────────────────────────────────────────────────

size_t ConvNet::flatFeatureSize() const noexcept
{
    return _curCh * _curLen;
}

size_t ConvNet::outputSize() const noexcept
{
    return _fc ? _fc->getOutputSize() : 0;
}

// ── Forward ───────────────────────────────────────────────────────────────────

const std::vector<double>& ConvNet::_forwardConv(const std::vector<double>& input)
{
    const std::vector<double>* cur = &input;
    for (auto& layer : _layers)
        cur = &layer->forward(*cur);
    return *cur;
}

std::vector<double> ConvNet::predict(const std::vector<double>& input)
{
    if (!_fc)
        throw std::logic_error("ConvNet::predict: call setFCHead() before predict()");
    const auto& flat = _forwardConv(input);
    _fc->setInputVector(flat);
    _fc->feedForward();
    std::vector<double> out;
    _fc->copyOutputVector(out);
    return out;
}

// ── Train (per-sample SGD) ────────────────────────────────────────────────────

double ConvNet::train(const std::vector<double>& input, const std::vector<double>& target)
{
    if (!_fc)
        throw std::logic_error("ConvNet::train: call setFCHead() before train()");

    // 1. Forward through conv/pool layers.
    const auto& flat = _forwardConv(input);

    // 2. Forward + backward through FC head.
    _fc->setInputVector(flat);
    _fc->feedForward();
    const double loss = _fc->calcMSE(target);
    _fc->backPropagate(target); // updates FC weights

    // 3. Get gradient w.r.t. the FC input (= flattened conv output).
    const Eigen::VectorXd ig = _fc->getInputGradient();
    std::vector<double> grad(ig.data(), ig.data() + ig.size());

    // 4. Backward through conv/pool layers in reverse.
    for (int i = static_cast<int>(_layers.size()) - 1; i >= 0; --i)
        grad = _layers[static_cast<size_t>(i)]->backward(grad, 0.0);

    return loss;
}

} // namespace nu
