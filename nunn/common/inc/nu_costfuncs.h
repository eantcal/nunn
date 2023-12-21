//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include "nu_vector.h"

namespace nu::cf {

//! Calculate the mean squared error of given
//! 'output vector' - 'target vector'
inline double calcMSE(Vector output, const Vector& target)
{
    output -= target;
    return 0.5 * output.euclideanNorm2();
}

//! Calculate the cross-entropy cost defined as
//! C=Sum(target*Log(output)+(1-target)*Log(1-output))/output.size()
double calcCrossEntropy(Vector output, const Vector& target);

using costfunc_t = double(Vector, const Vector&);

} // namespace nu::cf
