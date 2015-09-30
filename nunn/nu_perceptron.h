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
  This is an implementation of a Perceptron Neural Network which learns by example.
  
  You can give it examples of what you want the network to do and the algorithm 
  changes the network's weights. When training is finished, the net will give you 
  the required output for a particular input.
*/


/* -------------------------------------------------------------------------- */

#ifndef __NU_PERCEPTRON_H__
#define __NU_PERCEPTRON_H__


/* -------------------------------------------------------------------------- */

#include "nu_vector.h"
#include "nu_neuron.h"
#include "nu_stepf.h"
#include "nu_trainer.h"
#include "nu_noexcept.h"


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

//! This class represents a Perceptron neural net
class perceptron_t
{
public:
   using rvector_t = vector_t < double > ;

   static const char* ID_ANN;
   static const char* ID_NEURON;
   static const char* ID_INPUTS;

private:
   step_func_t _step_f;

public:
   enum class exception_t
   {
      size_mismatch,
      invalid_sstream_format
   };

   //! default ctor
   perceptron_t() = default;

   //! ctor
   perceptron_t(
      const size_t& n_of_inputs, 
      double learning_rate = 0.1,
      step_func_t step_f = step_func_t());
   

   //! Create a perceptron using data serialized into the given stream
   perceptron_t(std::stringstream& ss)
   {
      load(ss);
   }

   //! copy-ctor
   perceptron_t(const perceptron_t& nn) = default;


   //! move-ctor
   perceptron_t(perceptron_t&& nn) :
      _inputs_count(std::move(nn._inputs_count)),
      _learning_rate(std::move(nn._learning_rate)),
      _inputs(std::move(nn._inputs)),
      _neuron(std::move(nn._neuron))
   {
   }


   //! default assignement operator
   perceptron_t& operator=( const perceptron_t& nn ) = default;


   //! default assignement-move operator
   perceptron_t& operator=( perceptron_t&& nn ) 
   {
      if ( this != &nn )
      {
         _inputs_count = std::move(nn._inputs_count);
         _learning_rate = std::move(nn._learning_rate);
         _inputs = std::move(nn._inputs);
         _neuron = std::move(nn._neuron);
      }

      return *this;
   }


   //! Returns the number of inputs 
   size_t get_inputs_count() const NU_NOEXCEPT
   {
      return _inputs.size();
   }


   //! Returns current learning rate
   double get_learning_rate() const NU_NOEXCEPT
   {
      return _learning_rate;
   }


   //! Changes net learning rate
   void set_learning_rate(double new_rate)
   {
      _learning_rate = new_rate;
   }


   //! Sets net inputs
   void set_inputs(const rvector_t& inputs)
   {
      if (inputs.size() != _inputs.size())
         throw exception_t::size_mismatch;

      _inputs = inputs;
   }


   //! Get the net inputs
   void  get_inputs(rvector_t& inputs) const NU_NOEXCEPT
   {
      inputs = _inputs;
   }


   //! Get the net outputs 
   double get_output() const NU_NOEXCEPT
   {
      return _neuron.output;
   }


   //! Get the net outputs 
   double get_sharp_output() const NU_NOEXCEPT
   {
      return _step_f(_neuron.output);
   }


   //! Fire all neurons of the net and calculate the outputs
   void feed_forward();


   //! Fire the neuron, calculate the output
   //! and then apply the learing algorithm to the net
   void back_propagate(const double& target, double & output);

   
   //! Fire the neuron, calculate the output
   //! and then apply the learing algorithm to the net
   void back_propagate(const double& target)
   {
      double output;
      back_propagate(target, output);
   }


   //! Compute global error
   double error(const double& target) const NU_NOEXCEPT
   {
      return std::abs(target - get_output());
   }


   //! Build the net by using data of the given string stream
   std::stringstream& load(std::stringstream& ss);


   //! Save net status into the given string stream
   std::stringstream& save(std::stringstream& ss);


   //! Print the net state out to the given ostream
   std::ostream& dump(std::ostream& os);


   //! Build the net by using data of the given string stream
   friend std::stringstream& operator>>( 
      std::stringstream& ss, 
      perceptron_t& net )
   {
      return net.load(ss);
   }


   //! Save net status into the given string stream
   friend std::stringstream& operator<<( 
      std::stringstream& ss, 
      perceptron_t& net )
   {
      return net.save(ss);
   } 


   //! Print the net state out to the given ostream
   friend std::ostream& operator<<( std::ostream& os, perceptron_t& net )
   {
      return net.dump(os);
   }


   //! Reset all net weights using new random values
   void reshuffle_weights() NU_NOEXCEPT;

private:
   void _back_propagate(const double_t & target, const double_t& output);

   size_t _inputs_count;
   double _learning_rate = 0.1;
   rvector_t _inputs;
   neuron_t<double> _neuron;
};


/* -------------------------------------------------------------------------- */

//! The perceptron trainer class is a helper class for training perceptrons
class perceptron_trainer_t : 
   public nn_trainer_t<perceptron_t, nu::vector_t<double>, double>
{
public:
   perceptron_trainer_t( perceptron_t & nn, size_t epochs, double min_err) :
      nn_trainer_t<
         perceptron_t, 
         nu::vector_t<double>,
         double>(nn, epochs, min_err)
   {}
};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_PERCEPTRON_H__
