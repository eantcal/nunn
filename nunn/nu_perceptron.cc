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

perceptron_t::perceptron_t(const size_t& n_of_inputs, double learning_rate,
                           step_func_t step_f)
  : _inputs_count(n_of_inputs)
  , _learning_rate(learning_rate)
  , _step_f(step_f)
{
    if (n_of_inputs < 1)
        throw exception_t::size_mismatch;

    _inputs.resize(n_of_inputs, 0.0);
    _neuron.delta_weights.resize(n_of_inputs, 0.0);
    _neuron.weights.resize(n_of_inputs, 0.0);

    reshuffle_weights();
}


/* -------------------------------------------------------------------------- */

void perceptron_t::feed_forward() noexcept
{
    // For each layer (excluding input one) of neurons do...
    auto& neuron = _neuron;
    double sum = 0.0;

    // Sum of all the weights * input value
    for (size_t i = 0; i < _inputs.size(); ++i)
        sum += _inputs[i] * _neuron.weights[i];

    sum += _neuron.bias;

    neuron.output = sigmoid_t()(sum);
}


/* -------------------------------------------------------------------------- */

void perceptron_t::back_propagate(const double& target,
                                  double& output) noexcept
{
    // Calculate and get the outputs
    feed_forward();

    output = get_output();

    // Apply back_propagate algo
    _back_propagate(target, output);
}


/* -------------------------------------------------------------------------- */

void perceptron_t::_back_propagate(const double& target,
                                   const double& output) noexcept
{
    _neuron.error = (target - output);
    const double e = _learning_rate * _neuron.error;

    for (size_t i = 0; i < _inputs.size(); ++i)
        _neuron.weights[i] += e * _inputs[i];

    _neuron.bias += e;
}


/* -------------------------------------------------------------------------- */

std::stringstream& perceptron_t::load(std::stringstream& ss)
{
    std::string s;
    ss >> s;
    if (s != perceptron_t::ID_ANN)
        throw exception_t::invalid_sstream_format;

    ss >> _learning_rate;

    ss >> s;
    if (s != perceptron_t::ID_INPUTS)
        throw exception_t::invalid_sstream_format;

    ss >> _inputs;

    ss >> s;
    if (s != perceptron_t::ID_NEURON)
        throw exception_t::invalid_sstream_format;

    ss >> _neuron;

    return ss;
}


/* -------------------------------------------------------------------------- */

std::stringstream& perceptron_t::save(std::stringstream& ss) noexcept
{
    ss.clear();

    ss << perceptron_t::ID_ANN << std::endl;

    ss << _learning_rate << std::endl;

    ss << perceptron_t::ID_INPUTS << std::endl;
    ss << _inputs << std::endl;

    ss << perceptron_t::ID_NEURON << std::endl;
    ss << _neuron << std::endl;

    return ss;
}


/* -------------------------------------------------------------------------- */

void perceptron_t::reshuffle_weights() noexcept
{
    double weights_cnt = double(_neuron.weights.size());

    weights_cnt = std::sqrt(weights_cnt);

    // Initialize all the network weights
    // using random numbers within the range [-1,1]
    for (auto& w : _neuron.weights) {
        auto random_n = -1.0 + 2 * double(rand()) / double(RAND_MAX);
        w = random_n / weights_cnt;
    }

    for (auto& dw : _neuron.delta_weights)
        dw = 0;

    _neuron.bias = double(rand()) / double(RAND_MAX);
}


/* -------------------------------------------------------------------------- */

//! Print the net state out to the given ostream
std::ostream& perceptron_t::dump(std::ostream& os) noexcept
{
    os << "Perceptron " << std::endl;

    for (size_t in_idx = 0; in_idx < _neuron.weights.size(); ++in_idx) {
        os << "\t\tInput  [" << in_idx << "] = " << _inputs[in_idx]
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

const char* perceptron_t::ID_ANN = "perceptron";
const char* perceptron_t::ID_NEURON = "neuron";
const char* perceptron_t::ID_INPUTS = "inputs";


/* -------------------------------------------------------------------------- */

} // namespace nu
