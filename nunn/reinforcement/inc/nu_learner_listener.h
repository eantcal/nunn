//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include <cstddef>

namespace nu {

struct LearnerListener {
    virtual ~LearnerListener() = default;
    virtual bool notify(const double& reward, const size_t& move) = 0;
};

}
