//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_RNEURON_H__
#define __NU_RNEURON_H__


/* -------------------------------------------------------------------------- */

#include "nu_neuron.h"
#include <iostream>
#include <sstream>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

//! This class represents a neuron of a Recurrent NN layer
template <class T>
struct rneuron_t : public neuron_t<T>
{
    vector_t<T> delta_weights_tm1;

    friend std::stringstream& operator<<(std::stringstream& ss,
                                         const rneuron_t<T>& n) noexcept
    {
        const neuron_t<T>& bn = n;
        ss << bn;
        ss << n.delta_weights_tm1 << std::endl;

        return ss;
    }

    friend std::stringstream& operator>>(std::stringstream& ss,
                                         rneuron_t<T>& n) noexcept
    {
        neuron_t<T>& bn = n;
        ss >> bn;
        ss >> n.delta_weights_tm1;

        return ss;
    }

    void resize(size_t size) noexcept
    {
        neuron_t<T>::resize(size);
        delta_weights_tm1.resize(size);
    }
};


/* -------------------------------------------------------------------------- */
}


/* -------------------------------------------------------------------------- */

#endif // __NU_RNEURON_H__
