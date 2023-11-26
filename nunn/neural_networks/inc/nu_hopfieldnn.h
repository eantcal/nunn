//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once


#include "nu_random_gen.h"
#include "nu_stepf.h"
#include "nu_vector.h"

#include <list>


namespace nu {

//! This is an implementation of a Hopfield Neural Network
class HopfiledNN
{

  public:
    using FpVector = Vector<double>;

    enum class Exception
    {
        size_mismatch,
        invalid_sstream_format
    };


    //! default ctor
    HopfiledNN() = default;


    //! Create a net with pattern size equal to inputSize
    HopfiledNN(const size_t& inputSize) noexcept
      : _s(inputSize)
      , _w(inputSize * inputSize)
    {
    }


    //! Returns the capacity of the net
    size_t getCapacity() const noexcept
    {
        return size_t(0.138 * double(_s.size()));
    }


    //! Returns the number of patterns added to the net
    size_t getPatternsCount() const noexcept { return _patternSize; }


    //! Adds specified pattern
    void addPattern(const FpVector& input_pattern);


    //! Recall a pattern using as key the input one (it must be a vector
    //! containing [-1,1] values
    void recall(const FpVector& input_pattern, FpVector& output_pattern);


    //! Create a perceptron using data serialized into
    //! the given stream
    HopfiledNN(std::stringstream& ss) { load(ss); }

    //! copy-ctor
    HopfiledNN(const HopfiledNN& nn) = default;


    //! move-ctor
    HopfiledNN(HopfiledNN&& nn) noexcept
      : _s(std::move(nn._s))
      , _w(std::move(nn._w))
      , _patternSize(std::move(nn._patternSize))
    {
    }


    //! default assignment operator
    HopfiledNN& operator=(const HopfiledNN& nn) = default;


    //! default assignment-move operator
    HopfiledNN& operator=(HopfiledNN&& nn) noexcept = default;


    //! Returns the number of inputs
    size_t getInputSize() const noexcept { return _s.size(); }

    //! Build the net by using data of the given string stream
    std::stringstream& load(std::stringstream& ss);


    //! Save net status into the given string stream
    std::stringstream& save(std::stringstream& ss) noexcept;


    //! Print the net state out to the given ostream
    std::ostream& dump(std::ostream& os) noexcept;


    //! Build the net by using data of the given string stream
    friend std::stringstream& operator>>(std::stringstream& ss, HopfiledNN& net)
    {
        return net.load(ss);
    }


    //! Save net status into the given string stream
    friend std::stringstream& operator<<(std::stringstream& ss,
                                         HopfiledNN& net) noexcept
    {
        return net.save(ss);
    }


    //! Print the net state out to the given ostream
    friend std::ostream& operator<<(std::ostream& os, HopfiledNN& net) noexcept
    {
        return net.dump(os);
    }


    //! Reset the net status
    void clear() noexcept
    {
        _s = .0; // all zeros
        _w = .0; // all zeros
        _patternSize = 0;
    }

  private:
    static const char* ID_ANN;
    static const char* ID_WEIGHTS;
    static const char* ID_NEURON_ST;

    StepFunction step_f = StepFunction(0, -1, 1);

    void _propagate() noexcept;
    bool _propagateNeuron(size_t i) noexcept;

    FpVector _s; // neuron states
    FpVector _w; // weights matrix
    size_t _patternSize = 0;

    RandomGenerator<> _rndgen;
};


} // namespace nu
