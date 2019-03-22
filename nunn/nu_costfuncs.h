//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_COSTFUNCS_H__
#define __NU_COSTFUNCS_H__


/* -------------------------------------------------------------------------- */

#include "nu_vector.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

namespace cf {


/* --------------------------------------------------------------------------
 */

//! Calculate the mean squared error of given
//! 'output vector' - 'target vector'
inline double mean_squared_error(vector_t<double> output,
                                 const vector_t<double>& target)
{
    output -= target;
    return 0.5 * output.euclidean_norm2();
}


//! Calculate the cross-entropy cost defined as
//! C=Sum(target*Log(output)+(1-target)*Log(1-output))/output.size()
double cross_entropy(vector_t<double> output, const vector_t<double>& target);


/* --------------------------------------------------------------------------
 */

using costfunc_t = double(vector_t<double>, const vector_t<double>&);


/* --------------------------------------------------------------------------
 */

} // namespace cf


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_COSTFUNCS_H__
