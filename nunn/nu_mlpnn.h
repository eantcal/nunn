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
*  Author: Antonino Calderone <antonino.calderone@gmail.com>
*
*/

/**
  This is an implementation of a Artificial Neural Network which learns by
  example
  by using Back Propagation algorithm.
  You can give it examples of what you want the network to do and the algorithm
  changes the network's weights. When training is finished, the net will give
  you
  the required output for a particular input.

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
#include "nu_xmlpnn.h"

#include <utility>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

//! This class represents a MLP neural net
class mlp_neural_net_t : public xmlp_neural_net_t<neuron_t<double>>
{
  protected:
    using super_t = xmlp_neural_net_t<neuron_t<double>>;


    //! Called for serializing network status, returns NN id string
    const char* _get_id_ann() const noexcept override { return ID_ANN; }


    //! Called for serializing network status, returns neuron id string
    const char* _get_id_neuron() const noexcept override
    {
        return ID_NEURON;
    }


    //! Called for serializing network status, returns neuron-layer id string
    const char* _get_id_neuron_layer() const noexcept override
    {
        return ID_NEURON_LAYER;
    }


    //! Called for serializing network status, returns topology id string
    const char* _get_id_topology() const noexcept override
    {
        return ID_TOPOLOGY;
    }


    //! Called for serializing network status, returns inputs id string
    const char* _get_id_inputs() const noexcept override
    {
        return ID_INPUTS;
    }


  public:
    //! default ctor
    mlp_neural_net_t() = default;


    //! ctor
    mlp_neural_net_t(const topology_t& topology, double learning_rate = 0.1,
                     double momentum = 0.5, err_cost_t ec = err_cost_t::MSE);


    //! copy-ctor
    mlp_neural_net_t(const mlp_neural_net_t& nn) = default;


    //! move-ctor
    mlp_neural_net_t(mlp_neural_net_t&& nn)
      : super_t(nn)
    {
    }


    //! copy-assignment operator
    mlp_neural_net_t& operator=(const mlp_neural_net_t& nn) = default;


    //! move-assignment operator
    mlp_neural_net_t& operator=(mlp_neural_net_t&& nn)
    {
        super_t::operator=(std::move(nn));
        return *this;
    }

    //! Build the net by using data of given string stream
    friend std::stringstream& operator>>(std::stringstream& ss,
                                         mlp_neural_net_t& net)
    {
        return net.load(ss);
    }


    //! Save net status into given string stream
    friend std::stringstream& operator<<(std::stringstream& ss,
                                         mlp_neural_net_t& net)
    {
        return net.save(ss);
    }


    //! Dump the net status out to given ostream
    friend std::ostream& operator<<(std::ostream& os, mlp_neural_net_t& net)
    {
        return net.dump(os);
    }

    //! Reset all net weights using new random values
    void reshuffle_weights() noexcept;


  protected:
    //! This method is implemented in order to update
    //! network weights according to BP learning algorithm
    void _update_neuron_weights(neuron_t<double>& neuron,
                                size_t layer_idx) override;


  private:
    static const char* ID_ANN;
    static const char* ID_NEURON;
    static const char* ID_NEURON_LAYER;
    static const char* ID_TOPOLOGY;
    static const char* ID_INPUTS;
};


/* -------------------------------------------------------------------------- */

//! The trainer class is a helper class for MLP network training
class mlp_nn_trainer_t
  : public nn_trainer_t<mlp_neural_net_t, mlp_neural_net_t::rvector_t,
                        mlp_neural_net_t::rvector_t>
{
  public:
    mlp_nn_trainer_t(mlp_neural_net_t& nn, size_t epochs,
                     double min_err) noexcept
      : nn_trainer_t<mlp_neural_net_t, mlp_neural_net_t::rvector_t,
                     mlp_neural_net_t::rvector_t>(nn, epochs, min_err)
    {
    }
};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_MLPNN_H__
