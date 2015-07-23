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
*  Author: <antonino.calderone@ericsson.com>, <acaldmail@gmail.com>
*
*/

/* 
  This is an implementation of a Artificial Neural Network which learns by example
  by using Back Propagation algorithm.
  You can give it examples of what you want the network to do and the algorithm 
  changes the network�s weights. When training is finished, the net will give you 
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

#include "nu_vector.h"
#include "nu_sigmoid.h"

#include <vector>
#include <iostream>
#include <functional>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <cmath>


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

class mlp_neural_net_t
{
public:
   using rvector_t = vector_t < double > ;

   static const char* ID_ANN;
   static const char* ID_NEURON;
   static const char* ID_NEURON_LAYER;
   static const char* ID_TOPOLOGY;
   static const char* ID_INPUTS;

private:
   using actfunc_t = sigmoid_t;

   struct neuron_t
   {
      rvector_t weights;
      rvector_t delta_weights;
      double    bias = 0.0;
      double    output = 0.0;
      double    error = 0.0;

      friend 
      std::stringstream& operator<<(std::stringstream& ss, neuron_t& n)
      {
         ss << n.bias << std::endl;
         ss << n.weights << std::endl;
         ss << n.delta_weights << std::endl;
         return ss;
      }

      friend 
      std::stringstream& operator>>(std::stringstream& ss, neuron_t& n)
      {
         ss >> n.bias;
         ss >> n.weights;
         ss >> n.delta_weights << std::endl;
         return ss;
      }

   };

   using neuron_layer_t = std::vector < neuron_t > ;

public:
   using topology_t = vector_t < size_t > ;


   enum class exception_t
   {
      size_mismatch,
      invalid_sstream_format
   };


   mlp_neural_net_t() = default;


   mlp_neural_net_t(
      const topology_t& topology, 
      double learning_rate = 0.2,
      double momentum = 0.1);
   

   //! Create a network using data serialized into the given stream
   mlp_neural_net_t(std::stringstream& ss)
   {
      load(ss);
   }


   //! Returns the number of inputs 
   size_t get_inputs_count() const throw()
   {
      return _inputs.size();
   }


   //! Returns a const reference to topology vector
   const topology_t& get_topology() const throw( )
   {
      return _topology;
   }


   //! Returns current learning rate
   double get_learing_rate() const throw( )
   {
      return _learning_rate;
   }


   //! Change the learning rate of the net
   void set_learning_rate(double new_rate)
   {
      _learning_rate = new_rate;
   }


   //! Returns current momentum
   double get_momentum() const throw( )
   {
      return _momentum;
   }


   //! Change the momentum of the net
   void set_momentum(double new_momentum)
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
   const rvector_t& get_inputs() const throw()
   {
      return _inputs;
   }


   //! Get the net outputs 
   void get_outputs(rvector_t& outputs) throw();


   //! Fire all neurons of the net and calculate the outputs
   void feed_forward();


   //! Apply the Back Propagation Algorithm to the net
   void back_propagate(const rvector_t & target);


   //! Build the net by using data of the given string stream
   std::stringstream& load(std::stringstream& ss);


   //! Save net status into the given string stream
   std::stringstream& save(std::stringstream& ss);


   //! Print the net state out to the given ostream
   std::ostream& dump(std::ostream& os);


   //! Build the net by using data of the given string stream
   friend std::stringstream& operator>>( 
      std::stringstream& ss, 
      mlp_neural_net_t& net )
   {
      return net.load(ss);
   }


   //! Save net status into the given string stream
   friend std::stringstream& operator<<( 
      std::stringstream& ss, 
      mlp_neural_net_t& net )
   {
      return net.save(ss);
   } 


   //! Print the net state out to the given ostream
   friend std::ostream& operator<<( std::ostream& os, mlp_neural_net_t& net )
   {
      return net.dump(os);
   }


   //! Calculate the mean squared error of given 
   //! 'output vector' - 'target vector'
   static
   double mean_squared_error(rvector_t output, const rvector_t& target)
   {
      output -= target;
      return 0.5 * output.euclidean_norm2();
   }


   //! Calculate the mean squared error of net 'output vector' - 'target vector'
   double mean_squared_error(const rvector_t& target)
   {
      rvector_t output;
      get_outputs(output);
      return mean_squared_error(output, target);
   }


private:
   //Get input using layer index and input index
   //by means of layer==0 -> net inputs, 
   //layer>0 -> inputs == outputs of hidden neurons 
   //of previous layer 
   double _get_input(size_t layer, size_t idx) throw()
   {
      if (layer < 1)
         return _inputs[idx];

      const auto & neuron_layer = _neuron_layers[layer - 1];

      return neuron_layer[idx].output;
   }


   void _fire_neuron(
      neuron_layer_t & nlayer,
      size_t layer_idx,
      size_t out_idx) throw();


   static void _build(
      topology_t& topology,
      std::vector< neuron_layer_t >& neuron_layers,
      rvector_t & inputs );


   topology_t _topology;
   double _learning_rate = 0.1;
   double _momentum = 0.1;
   rvector_t _inputs;
   std::vector< neuron_layer_t > _neuron_layers;
};


/* -------------------------------------------------------------------------- */

} // namespace nu




/* -------------------------------------------------------------------------- */

#endif // __NU_MLPNN_H__
