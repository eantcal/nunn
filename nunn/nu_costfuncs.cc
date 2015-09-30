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
*  Author: <antonino.calderone@ericsson.com>, <acaldmail@gmail.com>
*
*/


/* -------------------------------------------------------------------------- */

#include "nu_costfuncs.h"


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

namespace cf
{

/* -------------------------------------------------------------------------- */

double cross_entropy(vector_t<double> output, const vector_t<double>& target)
{
   auto log_output = output;

   for ( auto & i : log_output )
      if ( i == 0.0 )
         i = 0.000001;

   log_output.log();

   vector_t<double> inv_target(target.size(), 1.0);
   inv_target -= target;

   vector_t<double> log_inv_output(output.size(), 1.0);
   log_inv_output -= output;

   for ( auto & i : log_inv_output )
      if ( i == 0.0 )
         i = 0.000001;

   log_inv_output.log();

   auto res = target;
   res *= log_output;

   inv_target *= log_inv_output;
   res += inv_target;

   return -res.sum() / double(res.size());
}


/* -------------------------------------------------------------------------- */

} // namespace cf


/* -------------------------------------------------------------------------- */

} // namespace nu
