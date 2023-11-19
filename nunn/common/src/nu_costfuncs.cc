//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#include "nu_costfuncs.h"
#include <limits>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

namespace cf {

/* -------------------------------------------------------------------------- */

double calcCrossEntropy(Vector<double> output, const Vector<double>& target)
{
    auto log_output = output;

    for (auto& i : log_output)
        if (i == 0.0)
            i = std::numeric_limits< double >::min();

    log_output.log();

    Vector<double> inv_target(target.size(), 1.0);
    inv_target -= target;

    Vector<double> log_inv_output(output.size(), 1.0);
    log_inv_output -= output;

    for (auto& i : log_inv_output)
        if (i == 0.0)
            i = std::numeric_limits< double >::min();

    log_inv_output.log();

    auto res = target;
    res *= log_output;

    inv_target *= log_inv_output;
    res += inv_target;

    return -res.sum() / double(res.size());
}


/* --------------------------------------------------------------------------
 */

} // namespace cf


/* -------------------------------------------------------------------------- */

} // namespace nu
