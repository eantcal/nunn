//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_costfuncs.h"
#include <limits>

namespace nu::cf {

double calcCrossEntropy(Vector output, const Vector& target)
{
    auto log_output = output;

    auto ensureNonZeros = [](auto&& v) {
        for (auto& i : v) {
            if (i == .0) {
                i = std::numeric_limits<double>::min();
            }
        }
    };

    ensureNonZeros(log_output);

    log_output.log();

    Vector inv_target(target.size(), 1.0);
    inv_target -= target;

    Vector log_inv_output(output.size(), 1.0);
    log_inv_output -= output;

    ensureNonZeros(log_inv_output);

    log_inv_output.log();

    auto res { target };
    res *= log_output;

    inv_target *= log_inv_output;
    res += inv_target;

    // Calculate the mean of the cross entropy values
    return -res.mean();
}

} // namespace nu::cf
