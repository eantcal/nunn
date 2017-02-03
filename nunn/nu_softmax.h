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

#ifndef __NU_SOFTMAX_H__
#define __NU_SOFTMAX_H__


/* -------------------------------------------------------------------------- */

#include <cmath>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template <class T=double>
class softmax_t {
public:
    using value_t = T;
    using pvector_t = std::vector<value_t>;
    using wvector_t = std::vector<value_t>;

    softmax_t(const value_t& temperature = value_t(1)) noexcept :
        _temp(temperature)
    {}

    pvector_t operator ()(const wvector_t& weights) {

        std::vector<double> probs(weights.size());

        value_t sum = 0;

        size_t i = 0;
        for (const auto& weight : weights) {
            const double pr = std::exp(weight / _temp);
            sum += pr;
            probs[i++] = pr;
        }

        for (auto& pr : probs) {
            pr /= sum;
        }

        // ok because of RVO
        return probs;
    }

private:
    value_t _temp = 1;
};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_SOFTMAX_H__

