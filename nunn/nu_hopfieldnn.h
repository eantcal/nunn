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

#ifndef __NU_HOPFIELDNN_H__
#define __NU_HOPFIELDNN_H__


/* -------------------------------------------------------------------------- */


#include "nu_stepf.h"
#include "nu_vector.h"

#include <list>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

//! This is an implementation of a Hopfield Neural Network
class hopfieldnn_t
{
  public:
    using rvector_t = vector_t<double>;

    enum class exception_t
    {
        size_mismatch,
        invalid_sstream_format
    };


    //! default ctor
    hopfieldnn_t() = default;


    //! Create a net with pattern size equal to n_of_inputs
    hopfieldnn_t(const size_t& n_of_inputs) noexcept
      : _s(n_of_inputs),
        _w(n_of_inputs* n_of_inputs)
    {
    }


    //! Returns the capacity of the net
    size_t get_capacity() const noexcept
    {
        return size_t(0.138 * double(_s.size()));
    }


    //! Returns the number of patterns added to the net
    size_t get_n_of_patterns() const noexcept { return _pattern_size; }


    //! Adds specified pattern
    void add_pattern(const rvector_t& input_pattern);


    //! Recall a pattern using as key the input one (it must be a vector
    //! containing [-1,1] values
    void recall(const rvector_t& input_pattern, rvector_t& output_pattern);


    //! Create a perceptron using data serialized into
    //! the given stream
    hopfieldnn_t(std::stringstream& ss) { load(ss); }

    //! copy-ctor
    hopfieldnn_t(const hopfieldnn_t& nn) = default;


    //! move-ctor
    hopfieldnn_t(hopfieldnn_t&& nn) noexcept
      : _s(std::move(nn._s)),
        _w(std::move(nn._w)),
        _pattern_size(std::move(_pattern_size))
    {
    }


    //! default assignment operator
    hopfieldnn_t& operator=(const hopfieldnn_t& nn) = default;


    //! default assignment-move operator
    hopfieldnn_t& operator=(hopfieldnn_t&& nn) noexcept;


    //! Returns the number of inputs
    size_t get_inputs_count() const noexcept { return _s.size(); }

    //! Build the net by using data of the given string stream
    std::stringstream& load(std::stringstream& ss);


    //! Save net status into the given string stream
    std::stringstream& save(std::stringstream& ss) noexcept;


    //! Print the net state out to the given ostream
    std::ostream& dump(std::ostream& os) noexcept;


    //! Build the net by using data of the given string stream
    friend std::stringstream& operator>>(std::stringstream& ss,
                                         hopfieldnn_t& net)
    {
        return net.load(ss);
    }


    //! Save net status into the given string stream
    friend std::stringstream& operator<<(std::stringstream& ss,
                                         hopfieldnn_t& net) noexcept
    {
        return net.save(ss);
    }


    //! Print the net state out to the given ostream
    friend std::ostream& operator<<(std::ostream& os,
                                    hopfieldnn_t& net) noexcept
    {
        return net.dump(os);
    }


    //! Reset the net status
    void clear() noexcept
    {
        for (auto& item : _s)
            item = 0;
        for (auto& item : _w)
            item = 0;
        _pattern_size = 0;
    }

  private:
    static const char* ID_ANN;
    static const char* ID_WEIGHTS;
    static const char* ID_NEURON_ST;

    step_func_t step_f = step_func_t(0, -1, 1);

    void _propagate() noexcept;
    bool _propagate_neuron(size_t i) noexcept;

    rvector_t _s; // neuron states
    rvector_t _w; // weights matrix
    size_t _pattern_size = 0;
};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_HOPFIELDNN_H__
