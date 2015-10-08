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

#ifndef __NU_NEURON_H__
#define __NU_NEURON_H__


/* -------------------------------------------------------------------------- */

#include "nu_vector.h"
#include <sstream>
#include <iostream>


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

//! This class represents a neuron of a neural net neuron layer
template<class T> struct neuron_t
{
   //! neuron weights
   vector_t < T > weights;

   //! amount by which weights will change
   vector_t < T > delta_weights;

   //! neuron bias
   T bias = T(0);

   //! neuron output
   T output = T(0);

   //! error field used by learning algorithm
   T error = T(0);

   //! Save neuron status into a given string stream
   friend std::stringstream&
      operator<<(std::stringstream& ss, const neuron_t<T>& n) NU_NOEXCEPT
   {
      ss << n.bias << std::endl;
      ss << n.weights << std::endl;
      ss << n.delta_weights << std::endl;

      return ss;
   }

   //! Load neuron status from a given string stream
   friend std::stringstream&
      operator>>(std::stringstream& ss, neuron_t<T>& n) NU_NOEXCEPT
   {
      ss >> n.bias;
      ss >> n.weights;
      ss >> n.delta_weights;

      return ss;
   }

   //! Resize both the weights and delta_weights vectors
   void resize(size_t size)
   {
      weights.resize(size);
      delta_weights.resize(size);
   }

};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_NEURON_H__

