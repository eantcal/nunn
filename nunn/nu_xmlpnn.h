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

#ifndef __NU_XMLPNN_H__
#define __NU_XMLPNN_H__


/* -------------------------------------------------------------------------- */

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


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

//! This class represents a base class for MLP and RMLP neural nets
template <class Neuron>
class xmlp_neural_net_t
{
  public:
    using rvector_t = vector_t<double>;

    using errv_func_t = std::function<void(const rvector_t& /* target */,
                                           const rvector_t& /* output */,
                                           rvector_t& /* result */)>;

    using cost_func_t = std::function<cf::costfunc_t>;


  protected:
    using actfunc_t = sigmoid_t;

    //! This class represents a neuron layer of a neural net.
    using neuron_layer_t = std::vector<Neuron>;

    cost_func_t _userdef_costf = nullptr;

  public:
    using topology_t = vector_t<size_t>;

    enum class err_cost_t
    {
        MSE,          //! mean square error cost function
        CROSSENTROPY, //! cross entropy cost function
        USERDEF
    };


    enum class exception_t
    {
        size_mismatch,
        invalid_sstream_format,
        userdef_costf_not_defined
    };


    //! default ctor
    xmlp_neural_net_t() = default;


    //! ctor
    xmlp_neural_net_t(const topology_t& topology, double learning_rate,
                      double momentum, err_cost_t ec) noexcept
      : _topology(topology),
        _learning_rate(learning_rate),
        _momentum(momentum),
        _err_cost_selector(ec)
    {
    }


    //! copy-ctor
    xmlp_neural_net_t(const xmlp_neural_net_t<Neuron>& nn) = default;


    //! move-ctor
    xmlp_neural_net_t(xmlp_neural_net_t<Neuron>&& nn) noexcept
      : _topology(std::move(nn._topology)),
        _learning_rate(std::move(nn._learning_rate)),
        _momentum(std::move(nn._momentum)),
        _inputs(std::move(nn._inputs)),
        _neuron_layers(std::move(nn._neuron_layers)),
        _err_cost_selector(std::move(nn._err_cost_selector))
    {
    }


    //! copy-assignment operator
    xmlp_neural_net_t& operator=(const xmlp_neural_net_t<Neuron>& nn) = default;


    //! move-assignment operator
    xmlp_neural_net_t& operator=(xmlp_neural_net_t<Neuron>&& nn) noexcept
    {
        if (this != &nn) {
            _topology = std::move(nn._topology);
            _learning_rate = std::move(nn._learning_rate);
            _momentum = std::move(nn._momentum);
            _inputs = std::move(nn._inputs);
            _neuron_layers = std::move(nn._neuron_layers);
            _err_cost_selector = std::move(nn._err_cost_selector);
        }

        return *this;
    }


    //! Selects the error cost function
    void select_error_cost_function(err_cost_t ec) noexcept
    {
        _err_cost_selector = ec;
    }


    //! Set a user defined cost function, selector is automatically
    //! set to err_cost_t::USERDEF;
    void set_error_cost_function(cost_func_t cf) noexcept
    {
        _err_cost_selector = err_cost_t::USERDEF;
        _userdef_costf = cf;
    }


    //! Get current error cost selector value
    err_cost_t get_err_cost() const noexcept { return _err_cost_selector; }


    //! Return the number of inputs
    size_t get_inputs_count() const noexcept { return _inputs.size(); }


    //! Return the number of outputs
    size_t get_outputs_count() const noexcept
    {
        if (_topology.empty())
            return 0;

        return _topology[_topology.size() - 1];
    }


    //! Return a const reference to topology vector
    const topology_t& get_topology() const noexcept { return _topology; }


    //! Return current learning rate
    double get_learning_rate() const noexcept { return _learning_rate; }


    //! Change the learning rate of the net
    void set_learning_rate(double new_rate) noexcept
    {
        _learning_rate = new_rate;
    }


    //! Return current momentum
    double get_momentum() const noexcept { return _momentum; }


    //! Change the momentum of the net
    void set_momentum(double new_momentum) noexcept
    {
        _momentum = new_momentum;
    }


    //! Set net inputs
    void set_inputs(const rvector_t& inputs)
    {
        if (inputs.size() != _inputs.size())
            throw exception_t::size_mismatch;

        _inputs = inputs;
    }


    //! Get the net inputs
    const rvector_t& get_inputs() const noexcept { return _inputs; }


    //! Get the net outputs
    void get_outputs(rvector_t& outputs) noexcept
    {
        const auto& last_layer = *_neuron_layers.crbegin();
        outputs.resize(last_layer.size());

        size_t idx = 0;
        for (const auto& neuron : last_layer)
            outputs[idx++] = neuron.output;
    }


    //! Fire all neurons of the net and calculate the outputs
    void feed_forward() noexcept
    {
        // For each layer (excluding input one) of neurons do...
        for (size_t layer_idx = 0; layer_idx < _neuron_layers.size();
             ++layer_idx) {
            auto& neuron_layer = _neuron_layers[layer_idx];

            const auto& size = neuron_layer.size();

            // Fire all neurons of this hidden / output layer
            for (size_t out_idx = 0; out_idx < size; ++out_idx)
                _fire_neuron(neuron_layer, layer_idx, out_idx);
        }
    }


    //! Fire all neurons of the net and calculate the outputs
    //! and then apply the Back Propagation Algorithm to the net
    virtual void back_propagate(const rvector_t& target_v, rvector_t& output_v)
    {
        // Calculate and get the outputs
        feed_forward();
        get_outputs(output_v);

        // Apply back_propagate algo
        _back_propagate(target_v, output_v);
    }


    //! Fire all neurons of the net and calculate the outputs
    //! and then apply the Back Propagation Algorithm to the net
    virtual void back_propagate(const rvector_t& target_v)
    {
        rvector_t output_v;

        // Calculate and get the outputs
        feed_forward();
        get_outputs(output_v);

        // Apply back_propagate algo
        _back_propagate(target_v, output_v);
    }


    //! Build the net by using data of the given string stream
    virtual std::stringstream& load(std::stringstream& ss)
    {
        std::string s;
        ss >> s;
        if (s != _get_id_ann())
            throw exception_t::invalid_sstream_format;

        ss >> _learning_rate;
        ss >> _momentum;

        ss >> s;
        if (s != _get_id_inputs())
            throw exception_t::invalid_sstream_format;

        ss >> _inputs;

        ss >> s;
        if (s != _get_id_topology())
            throw exception_t::invalid_sstream_format;

        ss >> _topology;

        _build(_topology, _neuron_layers, _inputs);

        for (auto& nl : _neuron_layers) {
            ss >> s;
            if (s != _get_id_neuron_layer())
                throw exception_t::invalid_sstream_format;

            for (auto& neuron : nl) {
                ss >> s;
                if (s != _get_id_neuron())
                    throw exception_t::invalid_sstream_format;

                ss >> neuron;
            }
        }

        return ss;
    }


    //! Save net status into the given string stream
    virtual std::stringstream& save(std::stringstream& ss) noexcept
    {
        ss.clear();

        ss << _get_id_ann() << std::endl;

        ss << _learning_rate << std::endl;
        ss << _momentum << std::endl;

        ss << _get_id_inputs() << std::endl;
        ss << _inputs << std::endl;

        ss << _get_id_topology() << std::endl;
        ss << _topology << std::endl;

        for (auto& nl : _neuron_layers) {
            ss << _get_id_neuron_layer() << std::endl;

            for (auto& neuron : nl) {
                ss << _get_id_neuron() << std::endl;
                ss << neuron << std::endl;
            }
        }

        return ss;
    }


    //! Print the net state out to the given ostream
    virtual std::ostream& dump(std::ostream& os) noexcept
    {
        os << "Net Inputs" << std::endl;
        size_t idx = 0;
        for (const auto& val : _inputs)
            os << "\t[" << idx++ << "] = " << val << std::endl;

        size_t layer_idx = 0;

        for (const auto& layer : _neuron_layers) {
            os << "\nNeuron layer " << layer_idx << " "
               << (layer_idx >= (_topology.size() - 2) ? "Output" : "Hidden")
               << std::endl;

            size_t neuron_idx = 0;

            for (const auto& neuron : layer) {
                os << "\tNeuron " << neuron_idx++ << std::endl;

                for (size_t in_idx = 0; in_idx < neuron.weights.size();
                     ++in_idx) {
                    os << "\t\tInput  [" << in_idx
                       << "] = " << _get_input(layer_idx, in_idx) << std::endl;

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


    //! Calculate mean squared error
    double mean_squared_error(const rvector_t& target)
    {
        rvector_t output;
        get_outputs(output);

        if (target.size() != output.size())
            throw exception_t::size_mismatch;

        return cf::mean_squared_error(output, target);
    }


    //! Calculate cross-entropy cost defined as
    //! C=(target*Log(output)+(1-target)*Log(1-output))/output.size()
    double cross_entropy(const rvector_t& target) noexcept
    {
        rvector_t output;
        get_outputs(output);

        if (target.size() != output.size())
            throw exception_t::size_mismatch;

        return cf::cross_entropy(output, target);
    }


    //! Calculate error cost
    virtual double calc_error_cost(const rvector_t& target)
    {
        rvector_t output;
        get_outputs(output);

        switch (_err_cost_selector) {
            case err_cost_t::USERDEF:
                if (!_userdef_costf)
                    throw exception_t::userdef_costf_not_defined;

                return _userdef_costf(output, target);

            case err_cost_t::CROSSENTROPY:
                return cf::cross_entropy(output, target);

            case err_cost_t::MSE:
            default:
                return cf::mean_squared_error(output, target);
        }
    }


    //! Return error vector function
    virtual errv_func_t get_errv_func() noexcept
    {
        switch (_err_cost_selector) {
            case err_cost_t::CROSSENTROPY:
                return _calc_xentropy_err_v;

            case err_cost_t::MSE:
            default:
                return _calc_mse_err_v;
        }
    }

  protected:
    //! Get input value for a neuron belonging to a given layer
    //! If layer is 0, it is related to input of the net
    double _get_input(size_t layer, size_t idx) noexcept
    {
        if (layer < 1)
            return _inputs[idx];

        const auto& neuron_layer = _neuron_layers[layer - 1];

        return neuron_layer[idx].output;
    }

    //! Fire all neurons of a given layer
    void _fire_neuron(neuron_layer_t& nlayer, size_t layer_idx,
                      size_t out_idx) noexcept
    {
        auto& neuron = nlayer[out_idx];

        double sum = 0.0;

        // Sum of all the weights * input value
        size_t idx = 0;
        for (const auto& wi : neuron.weights)
            sum += _get_input(layer_idx, idx++) * wi;

        sum += neuron.bias;

        neuron.output = actfunc_t()(sum);
    }


    // Called for serializing network status
    virtual const char* _get_id_ann() const noexcept = 0;
    virtual const char* _get_id_neuron() const noexcept = 0;
    virtual const char* _get_id_neuron_layer() const noexcept = 0;
    virtual const char* _get_id_topology() const noexcept = 0;
    virtual const char* _get_id_inputs() const noexcept = 0;


    //! This method must be implemented in order to update
    //! network weights according to the specific implementation
    //! of learning algorithm
    virtual void _update_neuron_weights(Neuron&, size_t) = 0;


    //! This method can be redefined in order to provide a
    //! specific implementation of learning algorithm
    virtual void _back_propagate(const rvector_t& target_v,
                                 const rvector_t& output_v)
    {
        if (target_v.size() != output_v.size())
            throw exception_t::size_mismatch;

        // -------- Calculate error for output neurons
        // --------------------------

        // Select the right error vector function
        auto errv_func = get_errv_func();

        // res_v = target - output
        rvector_t error_v;
        errv_func(target_v, output_v, error_v);

        // Copy error values into the output neurons
        size_t i = 0;
        for (auto& neuron : *_neuron_layers.rbegin())
            neuron.error = error_v[i++];


        // -------- Change output layer weights
        // ---------------------------------

        auto layer_idx = _topology.size() - 1;
        auto& layer = _neuron_layers[layer_idx - 1];

        for (size_t nidx = 0; nidx < layer.size(); ++nidx) {
            auto& neuron = layer[nidx];
            _update_neuron_weights(neuron, layer_idx);
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

            auto& h_layer = _neuron_layers[layer_idx - 1];

            // For each neuron of hidden layer
            for (size_t nidx = 0; nidx < h_layer.size(); ++nidx) {
                auto& neuron = h_layer[nidx];

                // Calculate error as output*(1-output)*s
                neuron.error = neuron.output * (1 - neuron.output);

                // where s = sum of w[nidx]*error of next layer neurons
                double sum = 0.0;

                const auto& nlsize = _neuron_layers[layer_idx].size();

                // For each neuron of next layer...
                for (size_t nnidx = 0; nnidx < nlsize; ++nnidx) {
                    auto& next_layer_neuron =
                      (_neuron_layers[layer_idx])[nnidx];

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

                _update_neuron_weights(neuron, layer_idx);
            }
        }
    }


    //! Initialize inputs and neuron layers of a net using a given
    //! topology
    static void _build(const topology_t& topology,
                       std::vector<neuron_layer_t>& neuron_layers,
                       rvector_t& inputs)
    {
        if (topology.size() < 3)
            throw(exception_t::size_mismatch);

        const size_t size = topology.size() - 1;

        neuron_layers.resize(size);

        size_t idx = 0;
        for (const auto& n_of_neurons : topology.to_stdvec()) {
            if (idx < 1) {
                inputs.resize(n_of_neurons);
            } else {
                auto& nl = neuron_layers[idx - 1];
                nl.resize(n_of_neurons);

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


    //! Calculate error vector in using MSE function
    static void _calc_mse_err_v(const rvector_t& target_v,
                                const rvector_t& outputs_v,
                                rvector_t& res_v) noexcept
    {
        // res = (1 - out) * out
        res_v.resize(outputs_v.size(), 1.0);
        res_v -= outputs_v;
        res_v *= outputs_v;

        // diff = target - out
        rvector_t diff_v(target_v);
        diff_v -= outputs_v;

        // Error vector = (1 - out) * out * (target - out)
        res_v *= diff_v;
    }


    //! Calculate error vector in using cross-entropy function
    static void _calc_xentropy_err_v(const rvector_t& target_v,
                                     const rvector_t& outputs_v,
                                     rvector_t& res_v) noexcept
    {
        // Error vector = target - out
        res_v = target_v;
        res_v -= outputs_v;
    }


    // Attributes
    topology_t _topology;
    double _learning_rate = 0.1;
    double _momentum = 0.1;
    rvector_t _inputs;
    std::vector<neuron_layer_t> _neuron_layers;
    err_cost_t _err_cost_selector = err_cost_t::MSE;
};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_XMLPNN_H__
