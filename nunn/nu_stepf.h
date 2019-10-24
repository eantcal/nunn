//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_STEPF_H__
#define __NU_STEPF_H__


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

class StepFunction {
public:
    StepFunction(double threshold = 0.0, 
                 double O_output = 0.0,
                 double I_output = 1.0) noexcept : 
        _threshold(threshold),
        _O_output(O_output),
        _I_output(I_output)
    {
    }

    double operator()(double x) const noexcept {
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
