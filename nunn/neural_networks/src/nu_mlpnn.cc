//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#include "nu_mlpnn.h"
#include "nu_random_gen.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

MlpNN::MlpNN(
    const Topology& topology,
    double learningRate, double momentum) :
     _topology(topology),
    _learningRate(learningRate),
    _momentum(momentum)
{
    _build(_topology, _neuronLayers, _inputVector);
    reshuffleWeights();
}


/* -------------------------------------------------------------------------- */

void MlpNN::_updateNeuronWeights(Neuron& neuron, size_t layerIdx)
{
    const auto lr_err = neuron.error * _learningRate;
    const auto m_err = neuron.error * _momentum;

    for (size_t inIdx = 0; inIdx < neuron.weights.size(); ++inIdx) {
        const auto dw_prev_step = neuron.deltaW[inIdx];

        neuron.deltaW[inIdx] =
          _getInput(layerIdx - 1, inIdx) * lr_err + m_err * dw_prev_step;

        neuron.weights[inIdx] += neuron.deltaW[inIdx];
    }

    neuron.bias = lr_err + m_err * neuron.bias;
}


/* -------------------------------------------------------------------------- */

void MlpNN::reshuffleWeights() noexcept
{
    double weights_cnt = 0.0;
    for (auto& nl : _neuronLayers)
        for (auto& neuron : nl)
            weights_cnt += double(neuron.weights.size());

    weights_cnt = std::sqrt(weights_cnt);

    RandomGenerator<> rndgen;

    // Initialize all the network weights
    // using random numbers within the range [-1,1]
    for (auto& nl : _neuronLayers) {
        for (auto& neuron : nl) {
            for (auto& w : neuron.weights) {
                auto random_n = -1.0 + 2 * rndgen();
                w = random_n / weights_cnt;
            }

            for (auto& dw : neuron.deltaW)
                dw = 0;

            neuron.bias = rndgen();
        }
    }
}


/* -------------------------------------------------------------------------- */

//! Get the net outputs
void MlpNN::copyOutputVector(FpVector& outputs) noexcept {
    const auto& last_layer = *_neuronLayers.crbegin();
    outputs.resize(last_layer.size());

    size_t idx = 0;
    for (const auto& neuron : last_layer)
        outputs[idx++] = neuron.output;
}


/* -------------------------------------------------------------------------- */

void MlpNN::feedForward() noexcept 
{
    // For each layer (excluding input one) of neurons do...
    for (size_t layerIdx = 0; layerIdx < _neuronLayers.size();
        ++layerIdx) {
        auto& neuronLayer = _neuronLayers[layerIdx];

        const auto& size = neuronLayer.size();

        // Fire all neurons of this hidden / output layer
        for (size_t outIdx = 0; outIdx < size; ++outIdx)
            _fireNeuron(neuronLayer, layerIdx, outIdx);
    }
}


/* -------------------------------------------------------------------------- */

void MlpNN::backPropagate(const FpVector& targetVector, FpVector& outputVector) 
{
    // Calculate and get the outputs
    feedForward();
    copyOutputVector(outputVector);

    // Apply backPropagate algo
    _backPropagate(targetVector, outputVector);
}


/* -------------------------------------------------------------------------- */

std::stringstream& MlpNN::load(std::stringstream& ss)
{
    std::string s;
    ss >> s;
    if (s != getNetId())
        throw Exception::invalid_sstream_format;

    ss >> _learningRate;
    ss >> _momentum;

    ss >> s;
    if (s != getInputVectorId())
        throw Exception::invalid_sstream_format;

    ss >> _inputVector;

    ss >> s;
    if (s != getTopologyId())
        throw Exception::invalid_sstream_format;

    ss >> _topology;

    _build(_topology, _neuronLayers, _inputVector);

    for (auto& nl : _neuronLayers) {
        ss >> s;
        if (s != getNeuronLayerId())
            throw Exception::invalid_sstream_format;

        for (auto& neuron : nl) {
            ss >> s;
            if (s != getNeuronId())
                throw Exception::invalid_sstream_format;

            ss >> neuron;
        }
    }

    return ss;
}


/* -------------------------------------------------------------------------- */

std::stringstream& MlpNN::save(std::stringstream& ss) noexcept
{
    ss.clear();

    ss << getNetId() << std::endl;

    ss << _learningRate << std::endl;
    ss << _momentum << std::endl;

    ss << getInputVectorId() << std::endl;
    ss << _inputVector << std::endl;

    ss << getTopologyId() << std::endl;
    ss << _topology << std::endl;

    for (auto& nl : _neuronLayers) {
        ss << getNeuronLayerId() << std::endl;

        for (auto& neuron : nl) {
            ss << getNeuronId() << std::endl;
            ss << neuron << std::endl;
        }
    }

    return ss;
}


/* -------------------------------------------------------------------------- */

std::ostream& MlpNN::formatJson(std::ostream& ss) noexcept
{
    ss.clear();

    ss << "{\"" << getNetId() << "\":{"<< std::endl;

    ss << "\"learningRate\":" << _learningRate << ",";
    ss << "\"momentum\":" << _momentum << ",";

    ss << "\"" << getInputVectorId() << "\":";
    _inputVector.formatJson(ss) << ",";

    ss << "\"" << getTopologyId() << "\":";
    _topology.formatJson(ss) << ",";
    ss << "\"layers\":{" << std::endl;

    size_t nlIdx = 0;
    for (auto& nl : _neuronLayers) {
        ss << "\"" << getNeuronLayerId() << nlIdx << "\":{" << std::endl;

        size_t neuronIdx = 0;
        for (auto& neuron : nl) {
            ss << "\"" << getNeuronId() << neuronIdx << "\":";
            
            neuron.formatJson(ss);

            if (++neuronIdx < nl.size()) {
                ss << ",";
            }
        }

        ss << "}";
        if (++nlIdx < _neuronLayers.size()) {
            ss << ",";
        }
        ss << std::endl;

    }

    ss << "}}}";

    return ss;
}


/* -------------------------------------------------------------------------- */

std::ostream& MlpNN::dump(std::ostream& os) noexcept
{
    os << "Net Inputs" << std::endl;
    size_t idx = 0;
    for (const auto& val : _inputVector)
        os << "\t[" << idx++ << "] = " << val << std::endl;

    size_t layerIdx = 0;

    for (const auto& layer : _neuronLayers) {
        os << "\nNeuron layer " << layerIdx << " "
            << (layerIdx >= (_topology.size() - 2) ? "Output" : "Hidden")
            << std::endl;

        size_t neuron_idx = 0;

        for (const auto& neuron : layer) {
            os << "\tNeuron " << neuron_idx++ << std::endl;

            for (size_t inIdx = 0; inIdx < neuron.weights.size();
                ++inIdx) {
                os << "\t\tInput  [" << inIdx
                    << "] = " << _getInput(layerIdx, inIdx) << std::endl;

                os << "\t\tWeight [" << inIdx
                    << "] = " << neuron.weights[inIdx] << std::endl;
            }

            os << "\t\tBias =       " << neuron.bias << std::endl;

            os << "\t\tOuput = " << neuron.output;
            os << std::endl;

            os << "\t\tError = " << neuron.error;
            os << std::endl;
        }

        ++layerIdx;
    }

    return os;
}


/* -------------------------------------------------------------------------- */

double MlpNN::calcMSE(const FpVector& targetVector)
{
    FpVector outputVector;
    copyOutputVector(outputVector);

    if (targetVector.size() != outputVector.size())
        throw Exception::size_mismatch;

    return cf::calcMSE(outputVector, targetVector);
}


/* -------------------------------------------------------------------------- */

double MlpNN::calcCrossEntropy(const FpVector& targetVector)
{
    FpVector outputVector;
    copyOutputVector(outputVector);

    if (targetVector.size() != outputVector.size())
        throw Exception::size_mismatch;

    return cf::calcCrossEntropy(outputVector, targetVector);
}


/* -------------------------------------------------------------------------- */

void MlpNN::_fireNeuron(
    NeuronLayer& nlayer, 
    size_t layerIdx,
    size_t outIdx) noexcept
{
    auto& neuron = nlayer[outIdx];

    double sum = 0.0;

    // Sum of all the weights * input value
    size_t idx = 0;
    for (const auto& wi : neuron.weights)
        sum += _getInput(layerIdx, idx++) * wi;

    sum += neuron.bias;

    neuron.output = Sigmoid()(sum);
}


/* -------------------------------------------------------------------------- */

void MlpNN::_backPropagate(const FpVector& targetVector,
    const FpVector& outputVector)
{
    if (targetVector.size() != outputVector.size())
        throw Exception::size_mismatch;

    // -------- Calculate error for output neurons
    // --------------------------

    // res_v = target - output
    FpVector error_v;
    _calcMSE(targetVector, outputVector, error_v);

    // Copy error values into the output neurons
    size_t i = 0;
    for (auto& neuron : *_neuronLayers.rbegin())
        neuron.error = error_v[i++];


    // -------- Change output layer weights
    // ---------------------------------

    auto layerIdx = _topology.size() - 1;
    auto& layer = _neuronLayers[layerIdx - 1];

    for (size_t nidx = 0; nidx < layer.size(); ++nidx) {
        auto& neuron = layer[nidx];
        _updateNeuronWeights(neuron, layerIdx);
    }


    // ------- Calculate hidden-layer errors and weights
    // --------------------
    //
    // Each hidden neuron error is given from its (output*(1-output))*s,
    // where s is the sum of next layer neurons error*weight of the
    // connection
    // between this hidden neuron and each layer neuron:
    //
    //                +-----+  W1  +----+           bias +--------+
    //                |  H  | ---- | N1 | E1         ----|        |
    //                +-----+ -    +----+           w1   | Neuron |
    //                        |                      ----|        |----
    //                        |    +----+                |        |
    //                        ---- | N2 | E2         ....|        |
    //                         W2  +----+                +--------+
    //                        .
    //                        .    ......
    //                        . . .. Nx . Ex
    //                             ......
    //
    //
    // Remark:
    // - output is output of H
    // - Wn is the weight of connection between H and next layers neuron
    // (Nn)
    // - errors are related to the next layer neurons output (Ex)

    while (layerIdx > 1) {
        --layerIdx;

        auto& hiddenLayer = _neuronLayers[layerIdx - 1];

        // For each neuron of hidden layer
        for (size_t nidx = 0; nidx < hiddenLayer.size(); ++nidx) {
            auto& neuron = hiddenLayer[nidx];

            // Calculate error as output*(1-output)*s
            neuron.error = neuron.output * (1 - neuron.output);

            // where s = sum of w[nidx]*error of next layer neurons
            double sum = 0.0;

            const auto& nlsize = _neuronLayers[layerIdx].size();

            // For each neuron of next layer...
            for (size_t nnidx = 0; nnidx < nlsize; ++nnidx) {
                auto& nextLayerNeuron =
                    (_neuronLayers[layerIdx])[nnidx];

                // ... add to the sum the product of its output error
                //     (as previusly computed)
                //     multiplied by the weights releated to neurons of
                //     hidden layer
                //     (they are related to hl-neuron index: nidx)
                sum +=
                    nextLayerNeuron.error * nextLayerNeuron.weights[nidx];

                // Add also bias-error rate
                if (nnidx == (nlsize - 1))
                    sum += nextLayerNeuron.error * nextLayerNeuron.bias;
            }

            neuron.error *= sum;

            _updateNeuronWeights(neuron, layerIdx);
        }
    }
}


/* -------------------------------------------------------------------------- */

void MlpNN::_build(
    const Topology& topology,
    std::vector<NeuronLayer>& neuronLayers,
    FpVector& inputs)
{
    if (topology.size() < 3)
        throw(Exception::size_mismatch);

    const size_t size = topology.size() - 1;

    neuronLayers.resize(size);

    size_t idx = 0;
    for (const auto& neuronsCount : topology.to_stdvec()) {
        if (idx < 1) {
            inputs.resize(neuronsCount);
        }
        else {
            auto& nl = neuronLayers[idx - 1];
            nl.resize(neuronsCount);

            // weights vector has more items than inputs
            // because ther is one implicit input used to
            // hold the bias
            for (auto& neuron : nl) {
                const auto size = topology[idx - 1];
                neuron.resize(size);
            }
        }

        ++idx;
    }
}


/* -------------------------------------------------------------------------- */

void MlpNN::_calcMSE(
    const FpVector& targetVector,
    const FpVector& outputVector,
    FpVector& res_v) noexcept
{
    // res = (1 - out) * out
    res_v.resize(outputVector.size(), 1.0);
    res_v -= outputVector;
    res_v *= outputVector;

    // diff = target - out
    FpVector diff_v(targetVector);
    diff_v -= outputVector;

    // Error vector = (1 - out) * out * (target - out)
    res_v *= diff_v;
}


/* -------------------------------------------------------------------------- */

} // namespace nu
