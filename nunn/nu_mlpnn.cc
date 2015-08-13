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


/* -------------------------------------------------------------------------- */

#include "nu_mlpnn.h"


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

void mlp_neural_net_t::_build(
   topology_t& topology,
   std::vector< neuron_layer_t >& neuron_layers,
   rvector_t & inputs
   )
{
   if ( topology.size() < 3 )
      throw( exception_t::size_mismatch );

   const size_t size = topology.size() - 1;

   neuron_layers.resize(size);

   size_t idx = 0;
   for ( const auto & n_of_neurons : topology )
   {
      if ( idx < 1 )
      {
         inputs.resize(n_of_neurons);
      }
      else
      {
         auto & nl = neuron_layers[idx - 1];
         nl.resize(n_of_neurons);

         // weights vector has more items than inputs
         // because ther is one implicit input used to
         // hold the bias 
         for ( auto & neuron : nl )
         {
            const auto size = topology[idx - 1];
            neuron.weights.resize(size);
            neuron.delta_weights.resize(size);
         }
      }

      ++idx;
   }
}


/* -------------------------------------------------------------------------- */

mlp_neural_net_t::mlp_neural_net_t(
   const topology_t& topology,  
   double learning_rate,
   double momentum) :
   _topology(topology),
   _learning_rate(learning_rate),
   _momentum(momentum)
{
   _build(_topology, _neuron_layers, _inputs);

   reshuffle_weights();
}


/* -------------------------------------------------------------------------- */

void mlp_neural_net_t::get_outputs(rvector_t& outputs) throw( )
{
   const auto & last_layer = *_neuron_layers.crbegin();
   outputs.resize(last_layer.size());

   size_t idx = 0;
   for ( const auto & neuron : last_layer )
      outputs[idx++] = neuron.output;
}


/* -------------------------------------------------------------------------- */

void mlp_neural_net_t::feed_forward()
{
   // For each layer (excluding input one) of neurons do...
   for ( size_t layer_idx = 0; layer_idx < _neuron_layers.size(); ++layer_idx )
   {
      auto & neuron_layer = _neuron_layers[layer_idx];

      const auto & size = neuron_layer.size();

      // Fire all neurons of this hidden / output layer
      for ( size_t out_idx = 0; out_idx < size; ++out_idx )
         _fire_neuron(neuron_layer, layer_idx, out_idx);
   }
}


/* -------------------------------------------------------------------------- */

void mlp_neural_net_t::back_propagate(const rvector_t & target)
{
   // Calculate the outputs
   feed_forward();


   // -------- Calculate error for output neurons --------------------------

   rvector_t outputs_v;
   get_outputs(outputs_v);

   if ( target.size() != outputs_v.size() )
      throw exception_t::size_mismatch;

   // res = (1 - out) * out 
   rvector_t res_v(outputs_v.size(), 1.0);
   res_v -= outputs_v;
   res_v *= outputs_v;

   // diff = target - out
   rvector_t diff_v(target);
   diff_v -= outputs_v;

   // Error vector = (1 - out) * out * (target - out)
   res_v *= diff_v;

   // Copy error values into the output neurons
   size_t i = 0;
   for ( auto & neuron : *_neuron_layers.rbegin() )
      neuron.error = res_v[i++];


   // -------- Change output layer weights ---------------------------------

   auto layer_idx = _topology.size() - 1;
   auto & layer = _neuron_layers[layer_idx - 1];

   for ( size_t nidx = 0; nidx < layer.size(); ++nidx )
   {
      auto & neuron = layer[nidx];

      for ( size_t in_idx = 0; in_idx < neuron.weights.size(); ++in_idx )
      {
         const auto dw_prev_step = neuron.delta_weights[in_idx];

         neuron.delta_weights[in_idx] =
            neuron.error * _get_input(layer_idx - 1, in_idx) * _learning_rate
            + _momentum * neuron.error * dw_prev_step;

         neuron.weights[in_idx] += 
            neuron.delta_weights[in_idx];
      }

      neuron.bias = neuron.error * _learning_rate + 
         neuron.bias * neuron.error * _momentum;
   }


   // ------- Calculate hidden-layer errors and weights --------------------
   //
   // Each hidden neuron error is given from its (output*(1-output))*s,
   // where s is the sum of next layer neurons error*weight of the connection 
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
   // - Wn is the weight of connection between H and next layers neuron (Nn)
   // - errors are related to the next layer neurons output (Ex)

   while ( layer_idx > 1 )
   {
      --layer_idx;

      auto & h_layer = _neuron_layers[layer_idx - 1];

      // For each neuron of hidden layer
      for ( size_t nidx = 0; nidx < h_layer.size(); ++nidx )
      {
         auto & neuron = h_layer[nidx];

         // Calculate error as output*(1-output)*s 
         neuron.error = neuron.output*( 1 - neuron.output );

         // where s = sum of w[nidx]*error of next layer neurons 
         double sum = 0.0;

         const auto & nlsize = _neuron_layers[layer_idx].size();

         // For each neuron of next layer...
         for ( size_t nnidx = 0; nnidx < nlsize; ++nnidx )
         {
            auto & next_layer_neuron = ( _neuron_layers[layer_idx] )[nnidx];

            // ... add to the sum the product of its output error (as previusly computed)
            //     multiplied by the weights releated to neurons of hidden layer 
            //     (they are related to hl-neuron index: nidx)
            sum += next_layer_neuron.error * next_layer_neuron.weights[nidx];

            //Add also bias-error rate
            if ( nnidx == ( nlsize - 1 ) )
               sum += next_layer_neuron.error * next_layer_neuron.bias;
         }

         neuron.error *= sum;

         for ( size_t in_idx = 0; in_idx < neuron.weights.size(); ++in_idx )
         {
            const auto dw_prev_step = neuron.delta_weights[in_idx];

            neuron.delta_weights[in_idx] =
               neuron.error * _get_input(layer_idx - 1, in_idx) * _learning_rate
               + _momentum * neuron.error * dw_prev_step;

            neuron.weights[in_idx] +=
               neuron.delta_weights[in_idx];
         }

         neuron.bias = neuron.error * _learning_rate +
            neuron.bias * neuron.error * _momentum;
      }
   }
}


/* -------------------------------------------------------------------------- */

void mlp_neural_net_t::_fire_neuron(
   neuron_layer_t & nlayer,
   size_t layer_idx,
   size_t out_idx) throw( )
{
   auto & neuron = nlayer[out_idx];

   double sum = 0.0;

   // Sum of all the weights * input value
   size_t idx = 0;
   for ( const auto & wi : neuron.weights )
      sum += _get_input(layer_idx, idx++) * wi;

   neuron.output = actfunc_t()( sum );
}


/* -------------------------------------------------------------------------- */

std::stringstream& mlp_neural_net_t::load(std::stringstream& ss)
{
   std::string s;
   ss >> s;
   if ( s != mlp_neural_net_t::ID_ANN )
      throw exception_t::invalid_sstream_format;

   ss >> _learning_rate;
   ss >> _momentum;

   ss >> s;
   if ( s != mlp_neural_net_t::ID_INPUTS )
      throw exception_t::invalid_sstream_format;

   ss >> _inputs;

   ss >> s;
   if ( s != mlp_neural_net_t::ID_TOPOLOGY )
      throw exception_t::invalid_sstream_format;

   ss >> _topology;

   mlp_neural_net_t::_build( _topology, _neuron_layers, _inputs);
   
   for ( auto & nl : _neuron_layers )
   {
      ss >> s;
      if ( s != mlp_neural_net_t::ID_NEURON_LAYER )
         throw exception_t::invalid_sstream_format;

      for ( auto & neuron : nl )
      {
         ss >> s;
         if ( s != mlp_neural_net_t::ID_NEURON )
            throw exception_t::invalid_sstream_format;

         ss >> neuron;
      }
   }

   return ss;
}


/* -------------------------------------------------------------------------- */

std::stringstream& mlp_neural_net_t::save(std::stringstream& ss)
{
   ss.clear();

   ss << mlp_neural_net_t::ID_ANN << std::endl;

   ss << _learning_rate << std::endl;
   ss << _momentum << std::endl;

   ss << mlp_neural_net_t::ID_INPUTS << std::endl;
   ss << _inputs << std::endl;

   ss << mlp_neural_net_t::ID_TOPOLOGY << std::endl;
   ss << _topology << std::endl;

   for ( auto & nl : _neuron_layers )
   {
      ss << mlp_neural_net_t::ID_NEURON_LAYER << std::endl;

      for ( auto & neuron : nl )
      {
         ss << mlp_neural_net_t::ID_NEURON << std::endl;
         ss << neuron << std::endl;
      }
   }

   return ss;
}


/* -------------------------------------------------------------------------- */

void mlp_neural_net_t::reshuffle_weights() throw( )
{
   // Initialize all the network weights 
   // using random numbers within the range [-1,1]
   for ( auto & nl : _neuron_layers )
   {
      for ( auto & neuron : nl )
      {
         for ( auto & w : neuron.weights )
            w = -1.0 + double(2 * rand()) / double(RAND_MAX);

         for ( auto & dw : neuron.delta_weights )
            dw = 0;

         neuron.bias = -1.0 + double(2 * rand()) / double(RAND_MAX);
      }
   }
}


/* -------------------------------------------------------------------------- */

//! Print the net state out to the given ostream
std::ostream& mlp_neural_net_t::dump(std::ostream& os)
{
   os << "Net Inputs" << std::endl;
   size_t idx = 0;
   for ( const auto & val : _inputs )
      os << "\t[" << idx++ << "] = " << val << std::endl;

   size_t layer_idx = 0;

   for ( const auto & layer : _neuron_layers )
   {
      os << "\nNeuron layer " << layer_idx
         << " "
         << ( layer_idx >= ( _topology.size() - 2 ) ? "Output" : "Hidden" )
         << std::endl;

      size_t neuron_idx = 0;

      for ( const auto & neuron : layer )
      {
         os << "\tNeuron " << neuron_idx++ << std::endl;

         for ( size_t in_idx = 0; in_idx < neuron.weights.size(); ++in_idx )
         {
            os << "\t\tInput  [" << in_idx << "] = "
               << _get_input(layer_idx, in_idx) << std::endl;

            os << "\t\tWeight [" << in_idx << "] = "
               << neuron.weights[in_idx] << std::endl;
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

const char* mlp_neural_net_t::ID_ANN = "ann";
const char* mlp_neural_net_t::ID_NEURON = "neuron";
const char* mlp_neural_net_t::ID_NEURON_LAYER = "layer";
const char* mlp_neural_net_t::ID_TOPOLOGY = "topology";
const char* mlp_neural_net_t::ID_INPUTS = "inputs";


/* -------------------------------------------------------------------------- */

} // namespace nu
