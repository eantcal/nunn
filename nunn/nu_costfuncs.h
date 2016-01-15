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

#ifndef __NU_COSTFUNCS_H__
#define __NU_COSTFUNCS_H__


/* -------------------------------------------------------------------------- */

#include "nu_vector.h"


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

namespace cf
{


/* -------------------------------------------------------------------------- */

//! Calculate the mean squared error of given 
//! 'output vector' - 'target vector'
inline double mean_squared_error(
   vector_t<double> output,
   const vector_t<double>& target)
{
   output -= target;
   return 0.5 * output.euclidean_norm2();
}


//! Calculate the cross-entropy cost defined as
//! C=Sum(target*Log(output)+(1-target)*Log(1-output))/output.size()
double cross_entropy(
   vector_t<double> output,
   const vector_t<double>& target);


/* -------------------------------------------------------------------------- */

using costfunc_t = double(vector_t<double>, const vector_t<double>&);


/* -------------------------------------------------------------------------- */

} // namespace cf


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_COSTFUNCS_H__
