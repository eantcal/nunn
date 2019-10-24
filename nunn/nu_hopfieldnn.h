//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


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
    using FpVector = Vector<double>;

    enum class Exception
    {
        size_mismatch,
        invalid_sstream_format
    };


    //! default ctor
    hopfieldnn_t() = default;


    //! Create a net with pattern size equal to inputSize
    hopfieldnn_t(const size_t& inputSize) noexcept
      : _s(inputSize),
        _w(inputSize* inputSize)
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
    void add_pattern(const FpVector& input_pattern);


    //! Recall a pattern using as key the input one (it must be a vector
    //! containing [-1,1] values
    void recall(const FpVector& input_pattern, FpVector& output_pattern);


    //! Create a perceptron using data serialized into
    //! the given stream
    hopfieldnn_t(std::stringstream& ss) { load(ss); }

    //! copy-ctor
    hopfieldnn_t(const hopfieldnn_t& nn) = default;


    //! move-ctor
    hopfieldnn_t(hopfieldnn_t&& nn) noexcept
      : _s(std::move(nn._s)),
        _w(std::move(nn._w)),
        _pattern_size(std::move(nn._pattern_size))
    {
    }


    //! default assignment operator
    hopfieldnn_t& operator=(const hopfieldnn_t& nn) = default;


    //! default assignment-move operator
    hopfieldnn_t& operator=(hopfieldnn_t&& nn) noexcept;


    //! Returns the number of inputs
    size_t getInputSize() const noexcept { return _s.size(); }

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

    StepFunction step_f = StepFunction(0, -1, 1);

    void _propagate() noexcept;
    bool _propagate_neuron(size_t i) noexcept;

    FpVector _s; // neuron states
    FpVector _w; // weights matrix
    size_t _pattern_size = 0;
};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_HOPFIELDNN_H__
