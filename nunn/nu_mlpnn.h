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
  changes the network's weights. When training is finished, the net will give you 
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


   enum class err_cost_t
   {
      MSE,          //! mean square error cost function
      CROSSENTROPY  //! cross entropy cost function
   };


   enum class exception_t
   {
      size_mismatch,
      invalid_sstream_format
   };


   mlp_neural_net_t() = default;


   mlp_neural_net_t(
      const topology_t& topology, 
      double learning_rate = 0.2,
      double momentum = 0.5);
   

   //! Create a network using data serialized into the given stream
   mlp_neural_net_t(std::stringstream& ss)
   {
      load(ss);
   }


   mlp_neural_net_t(const mlp_neural_net_t& nn) = default;

   mlp_neural_net_t(mlp_neural_net_t&& nn) :
      _topology(std::move(nn._topology)),
      _learning_rate(std::move(nn._learning_rate)),
      _momentum(std::move(nn._momentum)),
      _inputs(std::move(nn._inputs)),
      _neuron_layers(std::move(nn._neuron_layers))
   {
   }

   mlp_neural_net_t& operator=( const mlp_neural_net_t& nn ) = default;

   mlp_neural_net_t& operator=( mlp_neural_net_t&& nn ) 
   {
      if ( this != &nn )
      {
         _topology = std::move(nn._topology);
         _learning_rate = std::move(nn._learning_rate);
         _momentum = std::move(nn._momentum);
         _inputs = std::move(nn._inputs);
         _neuron_layers = std::move(nn._neuron_layers);
      }

      return *this;
   }
   


   //! Returns the number of inputs 
   size_t get_inputs_count() const throw()
   {
      return _inputs.size();
   }


   //! Returns the number of outputs 
   size_t get_outputs_count() const throw( )
   {
      if ( _topology.empty() )
         return 0;

      return _topology[_topology.size()-1];
   }
   

   //! Returns a const reference to topology vector
   const topology_t& get_topology() const throw( )
   {
      return _topology;
   }


   //! Returns current learning rate
   double get_learning_rate() const throw( )
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


   //! Fire all neurons of the net and calculate the outputs
   //! and then apply the Back Propagation Algorithm to the net
   void back_propagate(
      const rvector_t & target, 
      err_cost_t ec = err_cost_t::MSE)
   {
      rvector_t outputs_v;
      back_propagate(target, outputs_v, ec);
   }
   

   //! Fire all neurons of the net and calculate the outputs
   //! and then apply the Back Propagation Algorithm to the net
   void back_propagate(
      const rvector_t & target, 
      rvector_t& outputs,
      err_cost_t ec = err_cost_t::MSE);


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


   //! Calculate the cross-entropy cost defined as
   //! C=Sum(target*Log(output)+(1-target)*Log(1-output))/output.size()
   static 
   double cross_entropy(rvector_t output, const rvector_t& target);


   //! Calculate the mean squared error of net 'output vector' - 'target vector'
   double mean_squared_error(const rvector_t& target)
   {
      rvector_t output;
      get_outputs(output);
      return mean_squared_error(output, target);
   }


   //! Calculate the cross-entropy cost defined as
   //! C=Sum(target*Log(output)+(1-target)*Log(1-output))/output.size()
   double cross_entropy(const rvector_t& target)
   {
      rvector_t output;
      get_outputs(output);
      return cross_entropy(output, target);
   }


   //! Reset all net weights using new random values
   void reshuffle_weights() throw();

private:
   void _back_propagate(
      const rvector_t & target, 
      rvector_t& outputs,
      err_cost_t ec
      );

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

//! The trainer class is a helper class for network training
class mlp_nn_trainer_t
{
   friend class iterator;

public:
   struct iterator
   {
      friend class mlp_nn_trainer_t;
      mlp_nn_trainer_t * _trainer = nullptr;
      size_t _epoch = 0;

   private:
      iterator(mlp_nn_trainer_t & trainer, size_t epoch) throw( )
         : _trainer(&trainer),
         _epoch(epoch)
      {
      }

   public:
      iterator(iterator & it) throw( ) :
         _trainer(it._trainer),
         _epoch(it._epoch)
      {
      }

      iterator& operator=( iterator & it ) throw( )
      {
         if ( &it != this )
         {
            _trainer = it._trainer;
            _epoch = it._epoch;
         }

         return *this;
      }

      iterator(iterator && it) throw( ) :
         _trainer(std::move(it._trainer)),
         _epoch(std::move(it._epoch))
      {
      }

      iterator& operator=( iterator && it ) throw( )
      {
         if ( &it != this )
         {
            _trainer = std::move(it._trainer);
            _epoch = std::move(it._epoch);
         }

         return *this;
      }

      size_t get_epoch() const throw( )
      {
         return _epoch;
      }

      mlp_nn_trainer_t& operator*( ) const throw( )
      {
         return *_trainer;
      }

      mlp_nn_trainer_t* operator->( ) const throw( )
      {
         return _trainer;
      }

      iterator operator++( ) throw( )
      {
         ++_epoch;
         return *this;
      }

      iterator operator++( int ) throw( ) // post
      {
         iterator ret = *this;
         ++_epoch;
         return ret;
      }

      bool operator==( iterator & other ) const throw( )
      {
         return ( 
            _trainer == other._trainer && 
            _epoch == other._epoch );
      }

      bool operator!=( iterator & other ) const throw( )
      {
         return !this->operator==( other );
      }
   };


   iterator begin()
   {
      return iterator(*this, 0);
   }


   iterator end()
   {
      return iterator(*this, this->_epochs + 1);
   }


   mlp_nn_trainer_t(
      mlp_neural_net_t & nn,
      size_t epochs,
      double min_err,
      mlp_neural_net_t::err_cost_t err_cost
      ) :
      _nn(nn),
      _epochs(epochs),
      _min_err(min_err),
      _err_cost(err_cost),
      _err(0.0)
   {}


   bool train(
      const mlp_neural_net_t::rvector_t& input_vector,
      const mlp_neural_net_t::rvector_t& target_vector);


   size_t get_epochs() const throw()
   {
      return _epochs;
   }


   double get_min_err() const throw( )
   {
      return _min_err;
   }


   mlp_neural_net_t::err_cost_t get_err_cost() const throw( )
   {
      return _err_cost;
   }
   

   double get_error() const throw()
   {
      return _err;
   }

   private:
      nu::mlp_neural_net_t & _nn;
      size_t _epochs;
      double _min_err;
      mlp_neural_net_t::err_cost_t _err_cost;
      double _err;

};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_MLPNN_H__
