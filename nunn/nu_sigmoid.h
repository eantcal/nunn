//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


#ifndef __NU_SIGMOID_H__
#define __NU_SIGMOID_H__


/* -------------------------------------------------------------------------- */


#include "nu_vector.h"
#include <cmath>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

//! This class represents the logistic function
class sigmoid_t
{
  public:
    //! Calculate logistic function of x
    double operator()(double x) const noexcept
    {
        return (1 / (1 + exp(-x)));
    }

    //! Derive the logistic function in y
    static inline double derive(double y) noexcept { return (1 - y) * y; }
};


/* -------------------------------------------------------------------------- */

} // namespace


#endif // __NU_SIGMOID_H__
