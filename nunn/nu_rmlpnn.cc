//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#include "nu_rmlpnn.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

rmlp_neural_net_t::rmlp_neural_net_t(const topology_t& topology,
                                     double learning_rate, double momentum,
                                     err_cost_t ec)
  : super_t(topology, learning_rate, momentum, ec)
{
    _build(_topology, _neuron_layers, _inputs);
    reshuffle_weights();
}


/* -------------------------------------------------------------------------- */

void rmlp_neural_net_t::_update_neuron_weights(rneuron_t<double>& neuron,
                                               size_t layer_idx)
{
    const auto lr_err = neuron.error * _learning_rate;
    const auto m_err = neuron.error * _momentum;

    for (size_t in_idx = 0; in_idx < neuron.weights.size(); ++in_idx) {
        auto delta_weights_tm1 = neuron.delta_weights_tm1[in_idx];

        neuron.delta_weights_tm1[in_idx] = neuron.delta_weights[in_idx];

        neuron.delta_weights[in_idx] =
          _get_input(layer_idx - 1, in_idx) * lr_err +
          m_err * neuron.delta_weights_tm1[in_idx];

        auto& weights = neuron.weights[in_idx];

        weights += delta_weights_tm1;
        weights += neuron.delta_weights[in_idx];
    }

    neuron.bias = lr_err + m_err * neuron.bias;
}


/* -------------------------------------------------------------------------- */

void rmlp_neural_net_t::reshuffle_weights() noexcept
{
    double weights_cnt = 0.0;
    for (auto& nl : _neuron_layers)
        for (auto& neuron : nl)
            weights_cnt += double(neuron.weights.size());

    weights_cnt = std::sqrt(weights_cnt);

    // Initialize all the network weights
    // using random numbers within the range [-1,1]
    for (auto& nl : _neuron_layers) {
        for (auto& neuron : nl) {
            for (auto& w : neuron.weights) {
                auto random_n = -1.0 + 2 * double(rand()) / double(RAND_MAX);
                w = random_n / weights_cnt;
            }

            for (auto& dw : neuron.delta_weights_tm1)
                dw = 0;

            for (auto& dw : neuron.delta_weights)
                dw = 0;

            neuron.bias = double(rand()) / double(RAND_MAX);
        }
    }
}


/* -------------------------------------------------------------------------- */

const char* rmlp_neural_net_t::ID_ANN = "rmlp";
const char* rmlp_neural_net_t::ID_NEURON = "neuron";
const char* rmlp_neural_net_t::ID_NEURON_LAYER = "layer";
const char* rmlp_neural_net_t::ID_TOPOLOGY = "topology";
const char* rmlp_neural_net_t::ID_INPUTS = "inputs";


/* -------------------------------------------------------------------------- */

} // namespace nu
