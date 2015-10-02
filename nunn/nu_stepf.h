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

#ifndef __NU_STEPF_H__
#define __NU_STEPF_H__

#include "nu_noexcept.h"


/* -------------------------------------------------------------------------- */

namespace nu
{


/* -------------------------------------------------------------------------- */

class step_func_t
{
public:
   step_func_t(
      double threshold = 0.0,
      double O_output = 0.0,
      double I_output = 1.0) NU_NOEXCEPT
      :
      _threshold(threshold),
      _O_output(O_output),
      _I_output(I_output)
   {}

   double operator()(double x) const NU_NOEXCEPT
   {
      return (x > _threshold ? _I_output : _O_output);
   }

private:
   double _threshold;
   double _O_output;
   double _I_output;

};


/* -------------------------------------------------------------------------- */

} // namespace


#endif // __NU_STEPF_H__