//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_NEURON_H__
#define __NU_NEURON_H__


/* -------------------------------------------------------------------------- */

#include "nu_vector.h"
#include <iostream>
#include <sstream>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

//! This class represents a neuron of a neural net neuron layer
template <class T>
struct neuron_t
{
    //! neuron weights
    vector_t<T> weights;

    //! amount by which weights will change
    vector_t<T> delta_weights;

    //! neuron bias
    T bias = T(0);

    //! neuron output
    T output = T(0);

    //! error field used by learning algorithm
    T error = T(0);

    //! Save neuron status into a given string stream
    friend std::stringstream& operator<<(std::stringstream& ss,
                                         const neuron_t<T>& n) noexcept
    {
        ss << n.bias << std::endl;
        ss << n.weights << std::endl;
        ss << n.delta_weights << std::endl;

        return ss;
    }

    //! Load neuron status from a given string stream
    friend std::stringstream& operator>>(std::stringstream& ss,
                                         neuron_t<T>& n) noexcept
    {
        ss >> n.bias;
        ss >> n.weights;
        ss >> n.delta_weights;

        return ss;
    }

    //! Resize both the weights and delta_weights vectors
    void resize(size_t size) noexcept {
        weights.resize(size);
        delta_weights.resize(size);
    }
};


/* -------------------------------------------------------------------------- */
}


/* -------------------------------------------------------------------------- */

#endif // __NU_NEURON_H__
