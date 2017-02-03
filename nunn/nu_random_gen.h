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

#ifndef __NU_RANDOM_GEN_H__
#define __NU_RANDOM_GEN_H__


/* -------------------------------------------------------------------------- */

#include <random>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template <
    class T = double, 
    class E = std::mt19937,
    class D = std::uniform_real_distribution<T>
>
class random_gen_t {
public:
    using value_t = T;
    using engine_t = E;
    using distribution_t = D;

    random_gen_t(const value_t& min_value = 0,
                 const value_t& max_value = 1) : 
            _distribution(min_value, max_value) 
    {
        std::random_device rd;

        _engine.seed(rd());
    }

    value_t operator ()() {
        return _distribution(_engine);
    }

private:
    engine_t _engine;
    distribution_t _distribution;
};


/* -------------------------------------------------------------------------- */

}


#endif // __NU_RANDOM_GEN_H__

