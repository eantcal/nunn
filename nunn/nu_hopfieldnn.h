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

#ifndef __NU_HOPFIELDNN_H__
#define __NU_HOPFIELDNN_H__


/* -------------------------------------------------------------------------- */

#include "nu_vector.h"
#include "nu_stepf.h"
#include "nu_noexcept.h"

#include <list>


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

//! This is an implementation of a Hopfield Neural Network
class hopfieldnn_t
{
public:
   using rvector_t = vector_t < double >;

   static const char* ID_ANN;
   static const char* ID_NEURON;
   static const char* ID_INPUTS;

private:
   step_func_t step_f = step_func_t(0, -1, 1);

public:
   enum class exception_t
   {
      size_mismatch,
      invalid_sstream_format
   };


   //! default ctor
   hopfieldnn_t() = default;


   //! ctor
   hopfieldnn_t(const size_t& n_of_inputs) NU_NOEXCEPT :
      _s(n_of_inputs),
      _w(n_of_inputs * n_of_inputs)
   {
   }


   //! Returns the capacity of the net
   size_t get_capacity() const NU_NOEXCEPT
   {
      return size_t(0.138 * double(_s.size()));
   }


   //! Returns the number of patterns added to the net
   size_t get_n_of_patterns() const NU_NOEXCEPT
   {
      return _pattern_size;
   }


   //! Adds specified pattern 
   void add_pattern(const rvector_t& input_pattern)
   {
      const auto size = get_inputs_count();

      if (size != input_pattern.size())
         throw exception_t::size_mismatch;

      for (size_t i = 0; i < size; ++i)
         for (size_t j = 0; j < size; ++j)
         {
            if (i != j)
               _w[i * size + j] += 
                  input_pattern[i] * input_pattern[j];
         }
   }


   //! Recall a pattern using as key the input one (it must be a vector
   //! containing [-1,1] values
   void recall(const rvector_t& input_pattern, rvector_t& output_pattern)
   {
      if (get_inputs_count() != input_pattern.size())
         throw exception_t::size_mismatch;

      _s = input_pattern;
      _propagate();

      output_pattern = _s;
   }

#if 0
   //! Create a perceptron using data serialized into 
   //! the given stream
   hopfieldnn_t(std::stringstream& ss)
   {
      load(ss);
   }
#endif // TODO

   //! copy-ctor
   hopfieldnn_t(const hopfieldnn_t& nn) = default;


   //! move-ctor
   hopfieldnn_t(hopfieldnn_t&& nn) :
      _s(std::move(nn._s)),
      _w(std::move(nn._w)),
      _pattern_size(std::move(_pattern_size))
   {
   }


   //! default assignement operator
   hopfieldnn_t& operator=(const hopfieldnn_t& nn) = default;


   //! default assignement-move operator
   hopfieldnn_t& operator=(hopfieldnn_t&& nn)
   {
      if (this != &nn)
      {
         _s = std::move(nn._s);
         _w = std::move(nn._w);
         _pattern_size = std::move(_pattern_size);
      }

      return *this;
   }


   //! Returns the number of inputs 
   size_t get_inputs_count() const NU_NOEXCEPT
   {
      return _s.size();
   }

#if 0 // TODO

   //! Build the net by using data of the given string stream
   std::stringstream& load(std::stringstream& ss);


   //! Save net status into the given string stream
   std::stringstream& save(std::stringstream& ss);


   //! Print the net state out to the given ostream
   std::ostream& dump(std::ostream& os);


   //! Build the net by using data of the given string stream
   friend std::stringstream& operator>>(
      std::stringstream& ss,
      hopfieldnn_t& net)
   {
      return net.load(ss);
   }


   //! Save net status into the given string stream
   friend std::stringstream& operator<<(
      std::stringstream& ss,
      hopfieldnn_t& net)
   {
      return net.save(ss);
   }


   //! Print the net state out to the given ostream
   friend std::ostream& operator<<(std::ostream& os, hopfieldnn_t& net)
   {
      return net.dump(os);
   }
#endif

private:
   void _propagate()
   {
      size_t it = 0;
      size_t last_it = 0;

      do
      {
         ++it;
         size_t rnd_idx = rand() % get_inputs_count();

         if (_propagate_neuron(rnd_idx))
            last_it = it;

      } while (it - last_it < 10 * get_inputs_count());
   }


   bool _propagate_neuron(size_t i) NU_NOEXCEPT
   {
      bool changed = false;
      double sum = 0;

      const auto size = get_inputs_count();

      for (size_t j = 0; j < size; ++j)
         sum += _w[i*size + j] * _s[j];

      double state = 0.0;

      if (sum != 0.0)
      {
         if (sum < 0.0)
            state = -1;

         if (sum > 0.0)
            state = 1;

         if (state != _s[i])
         {
            changed = true;
            _s[i] = state;
         }
      }

      return changed;
   }

   rvector_t _s; // neuron states
   rvector_t _w; // weights matrix
   size_t _pattern_size = 0;
};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_HOPFIELDNN_H__