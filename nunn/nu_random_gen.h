//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


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
    using real_t = T;
    using engine_t = E;
    using distribution_t = D;

    random_gen_t(const real_t& min_value = 0,
                 const real_t& max_value = 1) : 
            _distribution(min_value, max_value) 
    {
        std::random_device rd;

        _engine.seed(rd());
    }

    real_t operator ()() {
        return _distribution(_engine);
    }

private:
    engine_t _engine;
    distribution_t _distribution;
};


/* -------------------------------------------------------------------------- */

}


#endif // __NU_RANDOM_GEN_H__

