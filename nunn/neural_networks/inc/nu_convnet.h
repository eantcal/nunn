//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// ConvNet: builder for 1D convolutional networks.
//
// Usage pattern:
//   nu::ConvNet cnn(1, 16);                        // 1 channel, 16 time steps
//   cnn.addConv1D(8, 5);                            // 8 filters, kernel=5
//   cnn.addMaxPool1D(4);                            // pool size 4
//   cnn.setFCHead({ LC(cnn.flatFeatureSize()),      // must match flatFeatureSize()
//                   LC(16, Activation::Tanh),
//                   LC(2,  Activation::Sigmoid) });
//   double loss = cnn.train(input, target);
//   auto   out  = cnn.predict(input);
//

#pragma once

#include "nu_conv.h"
#include "nu_mlpmatrixnn.h"

#include <memory>
#include <vector>

namespace nu {

class ConvNet {
public:
    // inChannels: number of input channels
    // inLength:   number of time steps (samples) per channel
    ConvNet(size_t inChannels, size_t inLength);

    // Append a Conv1DLayer.
    // lr: SGD learning rate for this layer's weights.
    ConvNet& addConv1D(
        size_t outChannels, size_t kernelSize, Activation act = Activation::ReLU, double lr = 0.01);

    // Append a MaxPool1DLayer.
    ConvNet& addMaxPool1D(size_t poolSize);

    // Set the fully-connected head. Must be called after all conv/pool layers.
    // fcLayers[0].size must equal flatFeatureSize().
    // lr: learning rate for the FC network.
    // Throws std::invalid_argument if the input size does not match.
    void setFCHead(const std::vector<MlpMatrixNN::LayerConfig>& fcLayers, double lr = 0.01);

    // Forward pass; returns FC head output.
    // Throws std::logic_error if setFCHead() has not been called.
    std::vector<double> predict(const std::vector<double>& input);

    // Forward + backward (per-sample SGD). Returns MSE loss.
    // Throws std::logic_error if setFCHead() has not been called.
    double train(const std::vector<double>& input, const std::vector<double>& target);

    // Size of the flattened feature vector after all conv/pool layers (= FC input size).
    size_t flatFeatureSize() const noexcept;

    // Size of each input sample (inChannels * inLength).
    size_t inputSize() const noexcept { return _inCh * _inLen; }

    // Number of outputs (from FC head).
    size_t outputSize() const noexcept;

private:
    size_t _inCh, _inLen;
    size_t _curCh, _curLen; // tracked as layers are appended

    std::vector<std::unique_ptr<IConvLayer1D>> _layers;
    std::unique_ptr<MlpMatrixNN> _fc;

    // Run conv/pool forward; returns reference to the last layer's output.
    const std::vector<double>& _forwardConv(const std::vector<double>& input);
};

} // namespace nu
