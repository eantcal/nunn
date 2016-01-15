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

#ifndef __NU_SIGMOID_H__
#define __NU_SIGMOID_H__


/* -------------------------------------------------------------------------- */

#include <cmath>
#include "nu_vector.h"
#include "nu_noexcept.h"


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

//! This class represents the logistic function
class sigmoid_t
{
public:

   //! Calculate logistic function of x
   double operator()(double x) const NU_NOEXCEPT
   {
      return (1 / (1 + exp(-x)));
   }

   //! Derive the logistic function in y
   static inline double derive(double y) NU_NOEXCEPT
   {
      return (1 - y) * y;
   }

};


/* -------------------------------------------------------------------------- */

} // namespace


#endif // __NU_SIGMOID_H__