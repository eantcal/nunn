//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#include "nu_perceptron.h"
#include "nu_sigmoid.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

Perceptron::Perceptron(const size_t& inputSize, double learningRate, StepFunction step_f)
  : _inputSize(inputSize)
  , _learningRate(learningRate)
  , _step_f(step_f)
{
    if (inputSize < 1)
        throw Exception::size_mismatch;

    _inputVector.resize(inputSize, 0.0);
    _neuron.deltaW.resize(inputSize, 0.0);
    _neuron.weights.resize(inputSize, 0.0);

    reshuffleWeights();
}


/* -------------------------------------------------------------------------- */

void Perceptron::feedForward() noexcept
{
    // For each layer (excluding input one) of neurons do...
    double sum = 0.0;

    // Sum of all the weights * input value
    for (size_t i = 0; i < _inputVector.size(); ++i)
        sum += _inputVector[i] * _neuron.weights[i];

    sum += _neuron.bias;

    _neuron.output = Sigmoid()(sum);
}


/* -------------------------------------------------------------------------- */

void Perceptron::runBackPropagationAlgo(const double& target, double& output) noexcept
{
    // Calculate and get the outputs
    feedForward();
    output = getOutput();

    // Apply runBackPropagationAlgo algo
    _neuron.error = (target - output);
    const double e = _learningRate * _neuron.error;

    for (size_t i = 0; i < _inputVector.size(); ++i)
        _neuron.weights[i] += e * _inputVector[i];

    _neuron.bias += e;
}


/* -------------------------------------------------------------------------- */

std::stringstream& Perceptron::load(std::stringstream& ss)
{
    std::string s;
    ss >> s;
    if (s != Perceptron::ID_ANN)
        throw Exception::invalid_sstream_format;

    ss >> _learningRate;

    ss >> s;
    if (s != Perceptron::ID_INPUTS)
        throw Exception::invalid_sstream_format;

    ss >> _inputVector;

    ss >> s;
    if (s != Perceptron::ID_NEURON)
        throw Exception::invalid_sstream_format;

    ss >> _neuron;

    return ss;
}


/* -------------------------------------------------------------------------- */

std::stringstream& Perceptron::save(std::stringstream& ss) noexcept
{
    ss.clear();

    ss << Perceptron::ID_ANN << std::endl;

    ss << _learningRate << std::endl;

    ss << Perceptron::ID_INPUTS << std::endl;
    ss << _inputVector << std::endl;

    ss << Perceptron::ID_NEURON << std::endl;
    ss << _neuron << std::endl;

    return ss;
}


/* -------------------------------------------------------------------------- */

void Perceptron::reshuffleWeights() noexcept
{
    double weights_cnt = double(_neuron.weights.size());

    weights_cnt = std::sqrt(weights_cnt);

    // Initialize all the network weights
    // using random numbers within the range [-1,1]
    for (auto& w : _neuron.weights) {
        auto random_n = -1.0 + 2 * double(rand()) / double(RAND_MAX);
        w = random_n / weights_cnt;
    }

    for (auto& dw : _neuron.deltaW)
        dw = 0;

    _neuron.bias = double(rand()) / double(RAND_MAX);
}


/* -------------------------------------------------------------------------- */

//! Print the net state out to the given ostream
std::ostream& Perceptron::dump(std::ostream& os) noexcept
{
    os << "Perceptron " << std::endl;

    for (size_t in_idx = 0; in_idx < _neuron.weights.size(); ++in_idx) {
        os << "\t\tInput  [" << in_idx << "] = " << _inputVector[in_idx]
           << std::endl;

        os << "\t\tWeight [" << in_idx << "] = " << _neuron.weights[in_idx]
           << std::endl;
    }

    os << "\t\tBias =       " << _neuron.bias << std::endl;

    os << "\t\tOuput = " << _neuron.output;
    os << std::endl;

    os << "\t\tError = " << _neuron.error;
    os << std::endl;

    return os;
}


/* -------------------------------------------------------------------------- */

} // namespace nu
