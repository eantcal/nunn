//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_LEARNER_LISTENER_H__
#define __NU_LEARNER_LISTENER_H__


/* -------------------------------------------------------------------------- */

#include <cstddef>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

struct LearnerListener
{
    virtual ~LearnerListener() {}
    virtual bool notify(const double& reward, const size_t& move) = 0;
};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_LEARNER_LISTENER_H__
