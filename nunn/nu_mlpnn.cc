/*
*  This file is part of nunnlib
*
*  nunnlib is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  nunnlib is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with nunnlib; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  US
*
*  Author: Antonino Calderone <acaldmail@gmail.com>
*
*/


/* -------------------------------------------------------------------------- */

#include "nu_mlpnn.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

mlp_neural_net_t::mlp_neural_net_t(const topology_t& topology,
    double learning_rate, double momentum, err_cost_t ec)
    : super_t(topology, learning_rate, momentum, ec)
{
    _build(_topology, _neuron_layers, _inputs);
    reshuffle_weights();
}


/* -------------------------------------------------------------------------- */

void mlp_neural_net_t::_update_neuron_weights(
    neuron_t<double>& neuron, size_t layer_idx)
{
    const auto lr_err = neuron.error * _learning_rate;
    const auto m_err = neuron.error * _momentum;

    for (size_t in_idx = 0; in_idx < neuron.weights.size(); ++in_idx) {
        const auto dw_prev_step = neuron.delta_weights[in_idx];

        neuron.delta_weights[in_idx]
            = _get_input(layer_idx - 1, in_idx) * lr_err + m_err * dw_prev_step;

        neuron.weights[in_idx] += neuron.delta_weights[in_idx];
    }

    neuron.bias = lr_err + m_err * neuron.bias;
}


/* -------------------------------------------------------------------------- */

void mlp_neural_net_t::reshuffle_weights() NU_NOEXCEPT
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

            for (auto& dw : neuron.delta_weights)
                dw = 0;

            neuron.bias = double(rand()) / double(RAND_MAX);
        }
    }
}


/* -------------------------------------------------------------------------- */

const char* mlp_neural_net_t::ID_ANN = "ann";
const char* mlp_neural_net_t::ID_NEURON = "neuron";
const char* mlp_neural_net_t::ID_NEURON_LAYER = "layer";
const char* mlp_neural_net_t::ID_TOPOLOGY = "topology";
const char* mlp_neural_net_t::ID_INPUTS = "inputs";


/* -------------------------------------------------------------------------- */

} // namespace nu
