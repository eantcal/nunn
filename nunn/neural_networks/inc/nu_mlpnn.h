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
 * @brief Implementation of an Artificial Neural Network using the Back Propagation algorithm.
 *
 * This file describes an implementation of an Artificial Neural Network (ANN)
 * that learns through the Back Propagation algorithm. The ANN adjusts its weights
 * based on provided input-output pairs during the training process, which continues
 * until the output error is minimized.
 *
 * The Back Propagation algorithm involves the following steps:
 * 1. Initializing the network's weights to small random numbers between -1 and +1.
 * 2. Applying inputs and calculating the output (forward pass).
 * 3. Calculating the error for each neuron (Target - Output).
 * 4. Adjusting the weights to reduce the overall error.
 */
// clang-format on

#pragma once

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

//! @class MlpNN
//! @brief Represents a Multi-Layer Perceptron (MLP) neural network.
class MlpNN {
public:
    using FpVector = Vector;
    using costFunction_t = std::function<cf::costfunc_t>;
    using NeuronLayer = std::vector<Neuron>;
    using Topology = std::vector<size_t>;

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

    //! Default constructor.
    MlpNN() = default;

    //! Constructor to initialize the neural network with topology, learning
    //! rate, and momentum.
    MlpNN(const Topology& topology,
        double learningRate = 0.1,
        double momentum = 0.5);

    // Default copy and move constructors.
    MlpNN(const MlpNN& nn) = default;
    MlpNN(MlpNN&& nn) noexcept = default;

    // Default copy and move assignment operators.
    MlpNN& operator=(const MlpNN& nn) = default;
    MlpNN& operator=(MlpNN&& nn) noexcept = default;

    //! Gets the number of inputs.
    size_t getInputSize() const noexcept;

    //! Gets the number of outputs.
    size_t getOutputSize() const noexcept;

    //! Returns a constant reference to the topology vector.
    const Topology& getTopology() const noexcept { return _topology; }

    //! Gets the current learning rate.
    double getLearningRate() const noexcept { return _learningRate; }

    //! Sets a new learning rate.
    void setLearningRate(double rate) noexcept { _learningRate = rate; }

    //! Gets the current momentum.
    double getMomentum() const noexcept { return _momentum; }

    //! Sets a new momentum.
    void setMomentum(double new_momentum) noexcept { _momentum = new_momentum; }

    // Sets the network's input vector.
    void setInputVector(const FpVector& inputs);

    //! Gets the network's input vector.
    const FpVector& getInputVector() const noexcept { return _inputVector; }

    //! Copies the content of the output vector to outputs.
    void copyOutputVector(FpVector& outputs) noexcept;

    //! Fires all neurons in the network and calculates the outputs (forward
    //! pass).
    void feedForward() noexcept;

    /**
     * @brief Applies the Back Propagation algorithm to adjust the network's
     * weights. It computes the gradient of the error and updates the weights to
     * minimize it.
     *
     * @param targetVector The desired output for the given inputs.
     * @param outputVector The actual output of the network which will be
     * adjusted.
     */
    void backPropagate(const FpVector& targetVector, FpVector& outputVector);

    //! Fires all neurons and applies the Back Propagation algorithm.
    void backPropagate(const FpVector& targetVector);

    //! Builds the network using data from the given string stream.
    std::stringstream& load(std::stringstream& ss);

    //! Saves the network's status into the given string stream.
    std::stringstream& save(std::stringstream& ss) noexcept;

    //! Formats the network's status as JSON into the given string stream.
    std::ostream& toJson(std::ostream& ss) noexcept;

    //! Dumps the network's state to the given output stream.
    std::ostream& dump(std::ostream& os) noexcept;

    //! Calculates the error between the target and output vectors.
    double calcError(const FpVector& targetVector);

    //! Calculates the cross-entropy cost.
    double calcCrossEntropy(const FpVector& targetVector);

    //! Build the net by using data of given string stream
    friend std::stringstream& operator>>(std::stringstream& ss, MlpNN& net)
    {
        return net.load(ss);
    }

    //! Save net status into given string stream
    friend std::stringstream& operator<<(std::stringstream& ss, MlpNN& net)
    {
        return net.save(ss);
    }

    //! Dump the net status out to given ostream
    friend std::ostream& operator<<(std::ostream& os, MlpNN& net)
    {
        return net.dump(os);
    }

    //! Resets all network weights using new random values.
    void reshuffleWeights() noexcept;

    // Serialization helper methods.
    constexpr std::string_view getNetId() const noexcept { return ID_ANN; }

    constexpr std::string_view getNeuronId() const noexcept
    {
        return ID_NEURON;
    }
    constexpr std::string_view getNeuronLayerId() const noexcept
    {
        return ID_NEURON_LAYER;
    }
    constexpr std::string_view getTopologyId() const noexcept
    {
        return ID_TOPOLOGY;
    }
    constexpr std::string_view getInputVectorId() const noexcept
    {
        return ID_INPUTS;
    }

private:
    // Updates the weights of a given neuron based on the Back Propagation
    // algorithm.
    void _updateNeuronWeights(Neuron& neuron, size_t layerIdx);

    // Retrieves the input value for a neuron in a specified layer.
    double _getInput(size_t layer, size_t idx) noexcept;

    // Activates all neurons in a specified layer.
    void _fireNeuron(NeuronLayer& nlayer,
        size_t layerIdx,
        size_t outIdx) noexcept;

    // Implements the Back Propagation algorithm.
    void _backPropagate(const FpVector& targetVector,
        const FpVector& outputVector);

    // Initializes the neural network using a specified topology.
    static void _build(const Topology& topology,
        std::vector<NeuronLayer>& neuronLayers,
        FpVector& inputs);

    // Calculates the error vector using the Mean Squared Error function.
    static void _calcMSE(const FpVector& targetVector,
        const FpVector& outputVector,
        FpVector& res_v) noexcept;

    costFunction_t _userdef_costf = nullptr; // User-defined cost function
    Topology _topology; // Network topology
    double _learningRate { 0.1 }; // Learning rate
    double _momentum { 0.1 }; // Momentum
    FpVector _inputVector; // Input vector
    std::vector<NeuronLayer> _neuronLayers; // Layers of neurons

    // Serialization identifiers
    constexpr static std::string_view ID_ANN { "ann" };
    constexpr static std::string_view ID_NEURON { "neuron" };
    constexpr static std::string_view ID_NEURON_LAYER { "layer" };
    constexpr static std::string_view ID_TOPOLOGY { "topology" };
    constexpr static std::string_view ID_INPUTS { "inputs" };
};

//! The trainer class is a helper class for MLP network training
struct MlpNNTrainer : public NNTrainer<MlpNN, MlpNN::FpVector, MlpNN::FpVector> {

    MlpNNTrainer(MlpNN& nn, size_t epochs, double minErr = -1) noexcept
        : NNTrainer<MlpNN, MlpNN::FpVector, MlpNN::FpVector>(nn, epochs, minErr)
    {
    }
};

} // namespace nu
