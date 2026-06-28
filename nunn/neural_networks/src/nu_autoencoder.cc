//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_autoencoder.h"

#include <stdexcept>

namespace nu {

// ── Validation ────────────────────────────────────────────────────────────────

static size_t validateAndGetLatent(const std::vector<size_t>& encoderSizes)
{
    if (encoderSizes.empty())
        throw std::invalid_argument("Autoencoder: encoderSizes must not be empty");
    return encoderSizes.back();
}

// ── Topology helpers ──────────────────────────────────────────────────────────

static std::vector<MlpMatrixNN::LayerConfig> fullTopology(
    size_t inputDim, const std::vector<size_t>& encoderSizes, Activation act)
{
    // [inputDim, e0, ..., bottleneck, ..., e0, inputDim]
    std::vector<MlpMatrixNN::LayerConfig> cfg;
    cfg.emplace_back(inputDim); // input descriptor (activation ignored)
    for (size_t s : encoderSizes)
        cfg.emplace_back(s, act);
    for (int i = static_cast<int>(encoderSizes.size()) - 2; i >= 0; --i)
        cfg.emplace_back(encoderSizes[static_cast<size_t>(i)], act);
    cfg.emplace_back(inputDim, Activation::Linear);
    return cfg;
}

static std::vector<MlpMatrixNN::LayerConfig> decoderTopology(
    size_t inputDim, const std::vector<size_t>& encoderSizes, Activation act)
{
    // [bottleneck, ..., e0, inputDim]
    std::vector<MlpMatrixNN::LayerConfig> cfg;
    cfg.emplace_back(encoderSizes.back()); // bottleneck = decoder input
    for (int i = static_cast<int>(encoderSizes.size()) - 2; i >= 0; --i)
        cfg.emplace_back(encoderSizes[static_cast<size_t>(i)], act);
    cfg.emplace_back(inputDim, Activation::Linear);
    return cfg;
}

// ── Construction ──────────────────────────────────────────────────────────────

Autoencoder::Autoencoder(
    size_t inputDim, std::vector<size_t> encoderSizes, Activation act, double lr)
    : _inputDim(inputDim)
    , _latentSize(validateAndGetLatent(encoderSizes)) // throws if empty, before _net/_decNet
    , _encoderLayerCount(encoderSizes.size())
    , _net(fullTopology(inputDim, encoderSizes, act), lr)
    , _decNet(decoderTopology(inputDim, encoderSizes, act), lr)
{
    _syncDecoder();
}

// ── Forward ───────────────────────────────────────────────────────────────────

std::vector<double> Autoencoder::encode(const std::vector<double>& x)
{
    _net.setInputVector(x);
    _net.feedForward();
    const Eigen::VectorXd& z = _net.getLayerOutput(_encoderLayerCount - 1);
    return std::vector<double>(z.data(), z.data() + z.size());
}

std::vector<double> Autoencoder::decode(const std::vector<double>& z)
{
    _decNet.setInputVector(z);
    _decNet.feedForward();
    std::vector<double> out;
    _decNet.copyOutputVector(out);
    return out;
}

std::vector<double> Autoencoder::reconstruct(const std::vector<double>& x)
{
    _net.setInputVector(x);
    _net.feedForward();
    std::vector<double> out;
    _net.copyOutputVector(out);
    return out;
}

double Autoencoder::reconstructionMSE(const std::vector<double>& x)
{
    _net.setInputVector(x);
    _net.feedForward();
    return _net.calcMSE(x);
}

// ── Training ──────────────────────────────────────────────────────────────────

double Autoencoder::train(const std::vector<std::vector<double>>& dataset, size_t epochs)
{
    double lastMSE = 0.0;
    for (size_t ep = 0; ep < epochs; ++ep) {
        double total = 0.0;
        for (const auto& x : dataset) {
            _net.setInputVector(x);
            _net.feedForward();
            _net.backPropagate(x); // target = input
            total += _net.calcMSE(x);
        }
        lastMSE = total / static_cast<double>(dataset.size());
    }
    _syncDecoder();
    return lastMSE;
}

// ── Weight management ─────────────────────────────────────────────────────────

void Autoencoder::reshuffleWeights()
{
    _net.reshuffleWeights();
    _syncDecoder();
}

void Autoencoder::_syncDecoder()
{
    const size_t n = _decNet.numLayers();
    for (size_t i = 0; i < n; ++i) {
        _decNet.setLayerW(i, _net.getLayerW(_encoderLayerCount + i));
        _decNet.setLayerB(i, _net.getLayerB(_encoderLayerCount + i));
    }
}

} // namespace nu
