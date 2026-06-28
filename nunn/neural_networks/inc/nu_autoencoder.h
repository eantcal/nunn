//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//
// Autoencoder built on top of MlpMatrixNN.
//
// Architecture:
//   Encoder: inputDim → e0 → e1 → ... → bottleneck
//   Decoder: bottleneck → ... → e1 → e0 → inputDim  (mirror of encoder)
//
// Training objective: minimise MSE(reconstruct(x), x)  — i.e., target = input.
//
// Usage:
//   nu::Autoencoder ae(784, {128, 32}, nu::Activation::Tanh, 0.005);
//   ae.train(dataset, 500);
//   auto z   = ae.encode(sample);   // latent vector (dim 32)
//   auto rec = ae.decode(z);        // reconstructed sample (dim 784)

#pragma once

#include "nu_activation.h"
#include "nu_mlpmatrixnn.h"

#include <vector>

namespace nu {

class Autoencoder {
public:
    // inputDim:     dimensionality of input and reconstructed output
    // encoderSizes: hidden layer sizes of the encoder half; last = bottleneck
    //               decoder = mirror (bottleneck → ... → inputDim)
    // act:          activation for all hidden layers; output uses Linear (MSE)
    // lr:           SGD learning rate applied to the full autoencoder network
    //
    // Throws std::invalid_argument if encoderSizes is empty.
    Autoencoder(size_t inputDim, std::vector<size_t> encoderSizes,
        Activation act = Activation::Tanh, double lr = 0.01);

    // Encode x → latent vector (bottleneck activation).
    std::vector<double> encode(const std::vector<double>& x);

    // Decode latent vector z → reconstructed input.
    // Uses decoder weights synced from the last train() or reshuffleWeights() call.
    std::vector<double> decode(const std::vector<double>& z);

    // encode then decode — equivalent to a full forward pass.
    std::vector<double> reconstruct(const std::vector<double>& x);

    // MSE between x and reconstruct(x).
    double reconstructionMSE(const std::vector<double>& x);

    // Train on dataset (one sample per entry; target = input) for `epochs` passes.
    // Returns mean reconstruction MSE over the final epoch.
    // Syncs decoder weights once after all epochs.
    double train(const std::vector<std::vector<double>>& dataset, size_t epochs);

    size_t getInputSize() const noexcept { return _inputDim; }
    size_t getLatentSize() const noexcept { return _latentSize; }
    double getLearningRate() const noexcept { return _net.getLearningRate(); }

    // Reinitialise all weights and sync decoder.
    void reshuffleWeights();

private:
    size_t _inputDim;
    size_t _latentSize;
    size_t _encoderLayerCount;

    MlpMatrixNN _net; // full autoencoder — trained end-to-end
    MlpMatrixNN _decNet; // decoder-only — weights synced from _net for decode()

    void _syncDecoder();
};

} // namespace nu
