//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include <random>

namespace nu {

template <class Engine = std::mt19937,
    class Distribution = std::uniform_real_distribution<double>>
struct RandomGenerator {
    RandomGenerator(const double& min_value = 0, const double& max_value = 1)
        : _distribution(min_value, max_value)
    {
        std::random_device rd;
        _engine.seed(rd());
    }

    double operator()() { return _distribution(_engine); }

private:
    Engine _engine;
    Distribution _distribution;
};

}
