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

MlpNN::MlpNN(const Topology& topology,
                                   double learningRate, double momentum) :
     _topology(topology),
    _learningRate(learningRate),
    _momentum(momentum)
{
    _build(_topology, _neuronLayers, _inputVector);
    reshuffleWeights();
}


/* -------------------------------------------------------------------------- */

void MlpNN::_updateNeuronWeights(Neuron& neuron, size_t layer_idx)
{
    const auto lr_err = neuron.error * _learningRate;
    const auto m_err = neuron.error * _momentum;

    for (size_t in_idx = 0; in_idx < neuron.weights.size(); ++in_idx) {
        const auto dw_prev_step = neuron.deltaW[in_idx];

        neuron.deltaW[in_idx] =
          _getInput(layer_idx - 1, in_idx) * lr_err + m_err * dw_prev_step;

        neuron.weights[in_idx] += neuron.deltaW[in_idx];
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
    for (size_t layer_idx = 0; layer_idx < _neuronLayers.size();
        ++layer_idx) {
        auto& neuronLayer = _neuronLayers[layer_idx];

        const auto& size = neuronLayer.size();

        // Fire all neurons of this hidden / output layer
        for (size_t out_idx = 0; out_idx < size; ++out_idx)
            _fireNeuron(neuronLayer, layer_idx, out_idx);
    }
}


/* -------------------------------------------------------------------------- */

void MlpNN::runBackPropagationAlgo(const FpVector& target_v, FpVector& output_v) 
{
    // Calculate and get the outputs
    feedForward();
    copyOutputVector(output_v);

    // Apply runBackPropagationAlgo algo
    _backPropagate(target_v, output_v);
}


/* -------------------------------------------------------------------------- */

void MlpNN::runBackPropagationAlgo(const FpVector& target_v)
{
    FpVector output_v;

    // Calculate and get the outputs
    feedForward();
    copyOutputVector(output_v);

    // Apply runBackPropagationAlgo algo
    _backPropagate(target_v, output_v);
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

std::ostream& MlpNN::dump(std::ostream& os) noexcept
{
    os << "Net Inputs" << std::endl;
    size_t idx = 0;
    for (const auto& val : _inputVector)
        os << "\t[" << idx++ << "] = " << val << std::endl;

    size_t layer_idx = 0;

    for (const auto& layer : _neuronLayers) {
        os << "\nNeuron layer " << layer_idx << " "
            << (layer_idx >= (_topology.size() - 2) ? "Output" : "Hidden")
            << std::endl;

        size_t neuron_idx = 0;

        for (const auto& neuron : layer) {
            os << "\tNeuron " << neuron_idx++ << std::endl;

            for (size_t in_idx = 0; in_idx < neuron.weights.size();
                ++in_idx) {
                os << "\t\tInput  [" << in_idx
                    << "] = " << _getInput(layer_idx, in_idx) << std::endl;

                os << "\t\tWeight [" << in_idx
                    << "] = " << neuron.weights[in_idx] << std::endl;
            }

            os << "\t\tBias =       " << neuron.bias << std::endl;

            os << "\t\tOuput = " << neuron.output;
            os << std::endl;

            os << "\t\tError = " << neuron.error;
            os << std::endl;
        }

        ++layer_idx;
    }

    return os;
}


/* -------------------------------------------------------------------------- */

double MlpNN::calcMSE(const FpVector& target)
{
    FpVector output;
    copyOutputVector(output);

    if (target.size() != output.size())
        throw Exception::size_mismatch;

    return cf::calcMSE(output, target);
}


/* -------------------------------------------------------------------------- */

double MlpNN::calcCrossEntropy(const FpVector& target)
{
    FpVector output;
    copyOutputVector(output);

    if (target.size() != output.size())
        throw Exception::size_mismatch;

    return cf::calcCrossEntropy(output, target);
}


/* -------------------------------------------------------------------------- */

void MlpNN::_fireNeuron(NeuronLayer& nlayer, size_t layer_idx,
    size_t out_idx) noexcept
{
    auto& neuron = nlayer[out_idx];

    double sum = 0.0;

    // Sum of all the weights * input value
    size_t idx = 0;
    for (const auto& wi : neuron.weights)
        sum += _getInput(layer_idx, idx++) * wi;

    sum += neuron.bias;

    neuron.output = Sigmoid()(sum);
}


/* -------------------------------------------------------------------------- */

void MlpNN::_backPropagate(const FpVector& target_v,
    const FpVector& output_v)
{
    if (target_v.size() != output_v.size())
        throw Exception::size_mismatch;

    // -------- Calculate error for output neurons
    // --------------------------

    // res_v = target - output
    FpVector error_v;
    _calcMSE(target_v, output_v, error_v);

    // Copy error values into the output neurons
    size_t i = 0;
    for (auto& neuron : *_neuronLayers.rbegin())
        neuron.error = error_v[i++];


    // -------- Change output layer weights
    // ---------------------------------

    auto layer_idx = _topology.size() - 1;
    auto& layer = _neuronLayers[layer_idx - 1];

    for (size_t nidx = 0; nidx < layer.size(); ++nidx) {
        auto& neuron = layer[nidx];
        _updateNeuronWeights(neuron, layer_idx);
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

    while (layer_idx > 1) {
        --layer_idx;

        auto& h_layer = _neuronLayers[layer_idx - 1];

        // For each neuron of hidden layer
        for (size_t nidx = 0; nidx < h_layer.size(); ++nidx) {
            auto& neuron = h_layer[nidx];

            // Calculate error as output*(1-output)*s
            neuron.error = neuron.output * (1 - neuron.output);

            // where s = sum of w[nidx]*error of next layer neurons
            double sum = 0.0;

            const auto& nlsize = _neuronLayers[layer_idx].size();

            // For each neuron of next layer...
            for (size_t nnidx = 0; nnidx < nlsize; ++nnidx) {
                auto& next_layer_neuron =
                    (_neuronLayers[layer_idx])[nnidx];

                // ... add to the sum the product of its output error
                //     (as previusly computed)
                //     multiplied by the weights releated to neurons of
                //     hidden layer
                //     (they are related to hl-neuron index: nidx)
                sum +=
                    next_layer_neuron.error * next_layer_neuron.weights[nidx];

                // Add also bias-error rate
                if (nnidx == (nlsize - 1))
                    sum += next_layer_neuron.error * next_layer_neuron.bias;
            }

            neuron.error *= sum;

            _updateNeuronWeights(neuron, layer_idx);
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
    const FpVector& target_v,
    const FpVector& outputs_v,
    FpVector& res_v) noexcept
{
    // res = (1 - out) * out
    res_v.resize(outputs_v.size(), 1.0);
    res_v -= outputs_v;
    res_v *= outputs_v;

    // diff = target - out
    FpVector diff_v(target_v);
    diff_v -= outputs_v;

    // Error vector = (1 - out) * out * (target - out)
    res_v *= diff_v;
}


/* -------------------------------------------------------------------------- */

} // namespace nu
