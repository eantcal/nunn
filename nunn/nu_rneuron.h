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
