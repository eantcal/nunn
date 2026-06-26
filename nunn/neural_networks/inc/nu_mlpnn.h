//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

// clang-format off
/**
 * @file nu_mlpnn.h
 *
 * @brief Multi-Layer Perceptron with configurable activations and cost function.
 *
 * Each hidden/output layer can have an independent activation function
 * (Sigmoid, Tanh, ReLU, LeakyReLU, Linear).  The cost function used during
 * back-propagation can be switched between MSE and Cross-Entropy at
 * construction time or via setCostFunction().
 *
 * Backward-compatible constructors accept the legacy plain topology vector
 * (all layers default to Sigmoid, cost function defaults to MSE).
 */
// clang-format on

#pragma once

#include "nu_activation.h"
#include "nu_costfuncs.h"
#include "nu_neuron.h"
#include "nu_sigmoid.h"
#include "nu_trainer.h"
#include "nu_vector.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace nu {

//! Cost functions available for back-propagation.
enum class CostFunction {
    MSE, //!< Mean Squared Error  (works with any output activation)
    CrossEntropy, //!< Binary cross-entropy (best paired with Sigmoid output)
};

//! @class MlpNN
//! @brief Multi-Layer Perceptron neural network.
class MlpNN {
public:
    using FpVector = Vector;
    using costFunction_t = std::function<cf::costfunc_t>;
    using NeuronLayer = std::vector<Neuron>;

    //! Plain topology: number of neurons per layer (input → hidden… → output).
    using Topology = std::vector<size_t>;

    //! Per-layer configuration: size + activation function.
    struct LayerConfig {
        size_t size{ 0 };
        Activation activation{ Activation::Sigmoid };

        LayerConfig() = default;
        LayerConfig(size_t s) noexcept
            : size(s)
        {
        }
        LayerConfig(size_t s, Activation a) noexcept
            : size(s)
            , activation(a)
        {
        }
    };

    // ── Exceptions ───────────────────────────────────────────────────────────

    class SizeMismatchException : public std::runtime_error {
    public:
        SizeMismatchException()
            : std::runtime_error("Size mismatch")
        {
        }
    };
    class InvalidSStreamFormatException : public std::runtime_error {
    public:
        InvalidSStreamFormatException()
            : std::runtime_error("Invalid stringstream format")
        {
        }
    };
    class UserdefCostfNotDefinedException : public std::runtime_error {
    public:
        UserdefCostfNotDefinedException()
            : std::runtime_error("User-defined cost function not defined")
        {
        }
    };

    // ── Constructors ─────────────────────────────────────────────────────────

    MlpNN() = default;

    //! Construct from plain topology — all layers get Sigmoid, cost = MSE.
    MlpNN(const Topology& topology, double learningRate = 0.1, double momentum = 0.5,
        CostFunction cf = CostFunction::MSE);

    //! Construct with per-layer activation and explicit cost function.
    //! layers[0].activation is ignored (input layer has no activation).
    MlpNN(const std::vector<LayerConfig>& layers, double learningRate = 0.1, double momentum = 0.5,
        CostFunction cf = CostFunction::MSE);

    MlpNN(const MlpNN&) = default;
    MlpNN(MlpNN&&) noexcept = default;
    MlpNN& operator=(const MlpNN&) = default;
    MlpNN& operator=(MlpNN&&) = default;

    // ── Getters / setters ────────────────────────────────────────────────────

    size_t getInputSize() const noexcept;
    size_t getOutputSize() const noexcept;

    const Topology& getTopology() const noexcept { return _topology; }

    //! Activation function for neuron layer i (0 = first hidden, last = output).
    Activation getLayerActivation(size_t i) const { return _layerActivations.at(i); }

    //! All per-layer activations (size == number of neuron layers).
    const std::vector<Activation>& getLayerActivations() const noexcept
    {
        return _layerActivations;
    }

    double getLearningRate() const noexcept { return _learningRate; }
    void setLearningRate(double r) noexcept { _learningRate = r; }

    double getMomentum() const noexcept { return _momentum; }
    void setMomentum(double m) noexcept { _momentum = m; }

    CostFunction getCostFunction() const noexcept { return _costFunction; }
    void setCostFunction(CostFunction cf) noexcept { _costFunction = cf; }

    void setInputVector(const FpVector& inputs);
    const FpVector& getInputVector() const noexcept { return _inputVector; }

    void copyOutputVector(FpVector& outputs) noexcept;

    // ── Forward / backward pass ───────────────────────────────────────────────

    void feedForward() noexcept;
    void backPropagate(const FpVector& targetVector, FpVector& outputVector);
    void backPropagate(const FpVector& targetVector);

    // ── Serialization ─────────────────────────────────────────────────────────

    std::stringstream& load(std::stringstream& ss);
    std::stringstream& save(std::stringstream& ss) noexcept;

    std::ostream& toJson(std::ostream& os) noexcept;
    std::istream& loadJson(std::istream& is);

    std::ostream& dump(std::ostream& os) noexcept;

    // ── Loss helpers ──────────────────────────────────────────────────────────

    double calcMSE(const FpVector& targetVector);
    double calcCrossEntropy(const FpVector& targetVector);

    // ── Stream operators ──────────────────────────────────────────────────────

    friend std::stringstream& operator>>(std::stringstream& ss, MlpNN& net) { return net.load(ss); }
    friend std::stringstream& operator<<(std::stringstream& ss, MlpNN& net) { return net.save(ss); }
    friend std::ostream& operator<<(std::ostream& os, MlpNN& net) { return net.dump(os); }

    void reshuffleWeights() noexcept;

    constexpr std::string_view getNetId() const noexcept { return ID_ANN; }
    constexpr std::string_view getNeuronId() const noexcept { return ID_NEURON; }
    constexpr std::string_view getNeuronLayerId() const noexcept { return ID_NEURON_LAYER; }
    constexpr std::string_view getTopologyId() const noexcept { return ID_TOPOLOGY; }
    constexpr std::string_view getInputVectorId() const noexcept { return ID_INPUTS; }

private:
    void _updateNeuronWeights(Neuron& neuron, size_t layerIdx);
    double _getInput(size_t layer, size_t idx) noexcept;
    void _fireNeuron(NeuronLayer& nlayer, size_t layerIdx, size_t outIdx) noexcept;
    void _backPropagate(const FpVector& targetVector, const FpVector& outputVector);

    static void _build(
        const Topology& topology, std::vector<NeuronLayer>& neuronLayers, FpVector& inputs);

    CostFunction _costFunction{ CostFunction::MSE };
    Topology _topology;
    std::vector<Activation> _layerActivations; //!< one per neuron layer
    double _learningRate{ 0.1 };
    double _momentum{ 0.1 };
    FpVector _inputVector;
    std::vector<NeuronLayer> _neuronLayers;

    constexpr static std::string_view ID_ANN{ "ann" };
    constexpr static std::string_view ID_NEURON{ "neuron" };
    constexpr static std::string_view ID_NEURON_LAYER{ "layer" };
    constexpr static std::string_view ID_TOPOLOGY{ "topology" };
    constexpr static std::string_view ID_INPUTS{ "inputs" };
};

//! Trainer helper for MLP networks.
struct MlpNNTrainer : public NNTrainer<MlpNN, MlpNN::FpVector, MlpNN::FpVector> {
    MlpNNTrainer(MlpNN& nn, size_t epochs, double minErr = -1) noexcept
        : NNTrainer<MlpNN, MlpNN::FpVector, MlpNN::FpVector>(nn, epochs, minErr)
    {
    }
};

} // namespace nu
