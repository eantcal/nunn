//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/**
    This is an implementation of a Artificial Neural Network which learns by
    example by using Back Propagation algorithm.
    You can give it examples of what you want the network to do and the algorithm
    changes the network's weights. When training is finished, the net will give
    you the required output for a particular input.

    Back Propagation algorithm
    1) Initializes the net by setting up all its weights to be small random
    numbers between -1 and +1.
    2) Applies input and calculates the output (forward pass).
    3) Calculates the Error of each neuron which is essentially Target-Output
    4) Changes the weights in such a way that the Error will get smaller

    Steps from 2 to 4 are repeated again and again until the Error is minimal
*/


/* -------------------------------------------------------------------------- */

#ifndef __NU_MLPNN_H__
#define __NU_MLPNN_H__


/* -------------------------------------------------------------------------- */

#include "nu_neuron.h"
#include "nu_costfuncs.h"

#include "nu_sigmoid.h"
#include "nu_trainer.h"
#include "nu_vector.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

#include <utility>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

//! This class represents a MLP neural net
class MlpNN {
public:
    using FpVector = Vector<double>;

    using costFunction_t = std::function<cf::costfunc_t>;

    //! This class represents a neuron layer of a neural net.
    using NeuronLayer = std::vector<Neuron>;

    //! NN topology
    using Topology = Vector<size_t>;

    //! List of execption errors
    enum class Exception {
        size_mismatch,
        invalid_sstream_format,
        userdef_costf_not_defined
    };

    //! default ctor
    MlpNN() = default;

    //! ctor
    MlpNN(
        const Topology& topology, 
        double learningRate = 0.1,
        double momentum = 0.5);

    //! copy-ctor
    MlpNN(const MlpNN& nn) = default;

    //! move-ctor
    MlpNN(MlpNN&& nn) noexcept
        : 
        _topology(std::move(nn._topology)),
        _learningRate(std::move(nn._learningRate)),
        _momentum(std::move(nn._momentum)),
        _inputVector(std::move(nn._inputVector)),
        _neuronLayers(std::move(nn._neuronLayers))
    {
    }

    //! copy-assignment operator
    MlpNN& operator=(const MlpNN& nn) = default;

    //! move-assignment operator
    MlpNN& operator=(MlpNN&& nn) noexcept {
        if (this != &nn) {
            _topology = std::move(nn._topology);
            _learningRate = std::move(nn._learningRate);
            _momentum = std::move(nn._momentum);
            _inputVector = std::move(nn._inputVector);
            _neuronLayers = std::move(nn._neuronLayers);
        }

        return *this;
    }

    //! Return the number of inputs
    size_t getInputSize() const noexcept { 
        return _inputVector.size(); 
    }

    //! Return the number of outputs
    size_t getOutputSize() const noexcept {
        if (_topology.empty())
            return 0;

        return _topology[_topology.size() - 1];
    }

    //! Return a const reference to topology vector
    const Topology& getTopology() const noexcept { 
        return _topology; 
    }

    //! Return current learning rate
    double getLearningRate() const noexcept {
        return _learningRate; 
    }

    //! Change the learning rate of the net
    void setLearningRate(double rate) noexcept {
        _learningRate = rate;
    }

    //! Return current momentum
    double getMomentum() const noexcept { 
        return _momentum; 
    }

    //! Change the momentum of the net
    void setMomentum(double new_momentum) noexcept {
        _momentum = new_momentum;
    }

    //! Set net inputs
    void setInputVector(const FpVector& inputs) {
        if (inputs.size() != _inputVector.size())
            throw Exception::size_mismatch;

        _inputVector = inputs;
    }

    //! Get the net inputs
    const FpVector& getInputVector() const noexcept {
        return _inputVector;
    }

    //! Copy content of output vector to outputs
    void copyOutputVector(FpVector& outputs) noexcept;

    //! Fire all neurons of the net and calculate the outputs
    void feedForward() noexcept;

    //! Fire all neurons of the net and calculate the outputs
    //! and then apply the Back Propagation Algorithm to the net
    void runBackPropagationAlgo(const FpVector& targetVector, FpVector& outputVector);

    //! Fire all neurons of the net and calculate the outputs
    //! and then apply the Back Propagation Algorithm to the net
    void runBackPropagationAlgo(const FpVector& targetVector) {
        FpVector outputVector; // dummy
        runBackPropagationAlgo(targetVector, outputVector);
    }

    //! Build the net by using data of the given string stream
    std::stringstream& load(std::stringstream& ss);

    //! Save net status into the given string stream
    std::stringstream& save(std::stringstream& ss) noexcept;

    //! Print the net state out to the given ostream
    std::ostream& dump(std::ostream& os) noexcept;
        
    //! Calculate mean squared error
    double calcMSE(const FpVector& target);

    //! Calculate cross-entropy cost defined as
    //! C=(target*Log(output)+(1-target)*Log(1-output))/output.size()
    double calcCrossEntropy(const FpVector& target);

    //! Build the net by using data of given string stream
    friend 
    std::stringstream& operator>>(std::stringstream& ss, MlpNN& net) {
        return net.load(ss);
    }

    //! Save net status into given string stream
    friend 
    std::stringstream& operator<<(std::stringstream& ss, MlpNN& net) {
        return net.save(ss);
    }

    //! Dump the net status out to given ostream
    friend 
    std::ostream& operator<<(std::ostream& os, MlpNN& net) {
        return net.dump(os);
    }

    //! Reset all net weights using new random values
    void reshuffleWeights() noexcept;

    //! Called for serializing network status, returns NN id string
    constexpr const char* getNetId() const noexcept {
        return ID_ANN;
    }

    // Called for serializing network status, returns neuron id string
    constexpr const char* getNeuronId() const noexcept {
        return ID_NEURON;
    }

    // Called for serializing network status, returns neuron-layer id string
    constexpr const char* getNeuronLayerId() const noexcept {
        return ID_NEURON_LAYER;
    }

    // Called for serializing network status, returns topology id string
    constexpr const char* getTopologyId() const noexcept {
        return ID_TOPOLOGY;
    }

    //! Called for serializing network status, returns inputs id string
    constexpr const char* getInputVectorId() const noexcept {
        return ID_INPUTS;
    }

private:
    // Update network weights according to BP learning algorithm
    void _updateNeuronWeights(Neuron& neuron, size_t layerIdx);

    // Get input value for a neuron belonging to a given layer
    // If layer is 0, it is related to input of the net
    double _getInput(size_t layer, size_t idx) noexcept {
        if (layer < 1)
            return _inputVector[idx];

        const auto& neuronLayer = _neuronLayers[layer - 1];

        return neuronLayer[idx].output;
    }

    // Fire all neurons of a given layer
    void _fireNeuron(NeuronLayer& nlayer, size_t layerIdx, size_t outIdx) noexcept;

    // Do back propagation
    void _backPropagate(const FpVector& targetVector, const FpVector& outputVector);

    // Initialize inputs and neuron layers of a net using a given topology
    static void _build(
        const Topology& topology,
        std::vector<NeuronLayer>& neuronLayers,
        FpVector& inputs);

    // Calculate error vector in using MSE function
    static void _calcMSE(
        const FpVector& targetVector,
        const FpVector& outputVector,
        FpVector& res_v) noexcept;
    

    // Attributes
    costFunction_t _userdef_costf = nullptr;
    Topology _topology;
    double _learningRate = 0.1;
    double _momentum = 0.1;
    FpVector _inputVector;
    std::vector<NeuronLayer> _neuronLayers;
    
    constexpr static const char* ID_ANN = "ann";
    constexpr static const char* ID_NEURON = "neuron";
    constexpr static const char* ID_NEURON_LAYER = "layer";
    constexpr static const char* ID_TOPOLOGY = "topology";
    constexpr static const char* ID_INPUTS = "inputs";
};


/* -------------------------------------------------------------------------- */

//! The trainer class is a helper class for MLP network training
struct MlpNNTrainer : public NNTrainer<MlpNN, MlpNN::FpVector, MlpNN::FpVector> {
    
    MlpNNTrainer(MlpNN& nn, size_t epochs, double minErr = -1) noexcept : 
        NNTrainer<MlpNN, MlpNN::FpVector, MlpNN::FpVector>(nn, epochs, minErr) {}

};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_MLPNN_H__
