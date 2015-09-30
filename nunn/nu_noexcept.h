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

#ifndef __NU_NOEXCEPT_H__
#define __NU_NOEXCEPT_H__


/* -------------------------------------------------------------------------- */

#if (defined(_MSC_VER) && _MSC_VER<=1800)
#define NU_NOEXCEPT throw()
#else
#define NU_NOEXCEPT noexcept
#endif


/* -------------------------------------------------------------------------- */

#endif //__NU_NOEXCEPT_H__