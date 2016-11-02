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

/*
This is an implementation of a Artificial Recurrent Neural Network
which learns by example by using Back Propagation Through Time
learning algorithm.

You can give it examples of what you want the network to do and the algorithm
changes the network's weights. When training is finished, the net will give you
the required output for a particular input.

BPTT algorithm is s the natural extension of
standard back-propagation used with MLP, which performs gradient descent
on a complete unfolded network.

The network training sequence starts at time t0 and ends at time t1,
the total cost function is simply the sum over time of the standard error
function at each time-step
*/


/* -------------------------------------------------------------------------- */

#ifndef __NU_RMLPNN_H__
#define __NU_RMLPNN_H__


/* -------------------------------------------------------------------------- */

#include "nu_rneuron.h"
#include "nu_xmlpnn.h"

#include <utility>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

//! This class represents a RMLP neural net
class rmlp_neural_net_t : public xmlp_neural_net_t<rneuron_t<double>>
{
  protected:
    using super_t = xmlp_neural_net_t<rneuron_t<double>>;


    //! Called for serializing network status, returns NN id string
    const char* _get_id_ann() const NU_NOEXCEPT override { return ID_ANN; }


    //! Called for serializing network status, returns neuron id string
    const char* _get_id_neuron() const NU_NOEXCEPT override
    {
        return ID_NEURON;
    }


    //! Called for serializing network status, returns neuron-layer id string
    const char* _get_id_neuron_layer() const NU_NOEXCEPT override
    {
        return ID_NEURON_LAYER;
    }


    //! Called for serializing network status, returns topology id string
    const char* _get_id_topology() const NU_NOEXCEPT override
    {
        return ID_TOPOLOGY;
    }


    //! Called for serializing network status, returns inputs id string
    const char* _get_id_inputs() const NU_NOEXCEPT override
    {
        return ID_INPUTS;
    }


  public:
    //! default ctor
    rmlp_neural_net_t() = default;


    //! ctor
    rmlp_neural_net_t(const topology_t& topology, double learning_rate = 0.1,
                      double momentum = 0.5, err_cost_t ec = err_cost_t::MSE);


    //! copy-ctor
    rmlp_neural_net_t(const rmlp_neural_net_t& nn) = default;


    //! move-ctor
    rmlp_neural_net_t(rmlp_neural_net_t&& nn)
      : super_t(nn)
    {
    }


    //! copy-assignment operator
    rmlp_neural_net_t& operator=(const rmlp_neural_net_t& nn) = default;


    //! move-assignment operator
    rmlp_neural_net_t& operator=(rmlp_neural_net_t&& nn)
    {
        super_t::operator=(std::move(nn));
        return *this;
    }


    //! Build the net by using data of the given string stream
    friend std::stringstream& operator>>(std::stringstream& ss,
                                         rmlp_neural_net_t& net)
    {
        return net.load(ss);
    }


    //! Save net status into the given string stream
    friend std::stringstream& operator<<(std::stringstream& ss,
                                         rmlp_neural_net_t& net)
    {
        return net.save(ss);
    }


    //! Print the net state out to the given ostream
    friend std::ostream& operator<<(std::ostream& os, rmlp_neural_net_t& net)
    {
        return net.dump(os);
    }


    //! Reset all net weights using new random values
    void reshuffle_weights() NU_NOEXCEPT;


  protected:
    //! This method is implemented in order to update
    //! network weights according to BPTT learning algorithm
    void _update_neuron_weights(rneuron_t<double>& neuron,
                                size_t layer_idx) override;

    std::stringstream& _load(std::stringstream& ss);


  private:
    static const char* ID_ANN;
    static const char* ID_NEURON;
    static const char* ID_NEURON_LAYER;
    static const char* ID_TOPOLOGY;
    static const char* ID_INPUTS;
};


/* -------------------------------------------------------------------------- */

//! The trainer class is a helper class for network training
class rmlp_nn_trainer_t
  : public nn_trainer_t<rmlp_neural_net_t, rmlp_neural_net_t::rvector_t,
                        rmlp_neural_net_t::rvector_t>
{
  public:
    rmlp_nn_trainer_t(rmlp_neural_net_t& nn, size_t epochs, double min_err)
      : nn_trainer_t<rmlp_neural_net_t, rmlp_neural_net_t::rvector_t,
                     rmlp_neural_net_t::rvector_t>(nn, epochs, min_err)
    {
    }
};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_RMLPNN_H__
