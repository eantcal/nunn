//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#include "nu_qmtx.h"

#include <vector>
#include <cassert>
#include <exception>

namespace nu {


/* -------------------------------------------------------------------------- */

qmtx_t::qmtx_t(size_t n_of_states) 
{
    _data.resize(n_of_states);
    for (auto & row : _data) {
        row.resize(n_of_states);
    }
}


/* -------------------------------------------------------------------------- */

void qmtx_t::fill(const double& value) noexcept {
    for (auto & row : data()) {
        for (auto & v : row) {
            v = value;
        }
    }
}


/* -------------------------------------------------------------------------- */

double qmtx_t::max(size_t rowidx) const {
    size_t maxidx = 0;
    double maxvalue = 0;
    _max(rowidx, maxidx, maxvalue);
    return maxvalue;
}


/* -------------------------------------------------------------------------- */

size_t qmtx_t::maxarg(size_t rowidx) const {
    size_t maxidx = 0;
    double maxvalue = 0;
    _max(rowidx, maxidx, maxvalue);
    return maxidx;
}


/* -------------------------------------------------------------------------- */

void qmtx_t::normalize() {
    bool ft = true;
    double max = 0;

    for (auto & row : data()) {
        for (auto & v : row) {
            if (ft) {
                max = v;
                ft = false;
            }
            else if (v > max) {
                max = v;
            }
        }
    }

    if (max != 0) for (auto & row : data()) {
        for (auto & v : row) {
            v /= (max / 100.0);
        }
    }
}


/* -------------------------------------------------------------------------- */

qmtx_t::vect_t & qmtx_t::operator[](const size_t& rowidx) {
    if (rowidx >= size()) {
        assert(0);
        throw exception_t::invalid_index;
    }

    return data()[rowidx];
}


/* -------------------------------------------------------------------------- */

const qmtx_t::vect_t & qmtx_t::operator[](const size_t& rowidx) const {
    if (rowidx >= size()) {
        assert(0);
        throw exception_t::invalid_index;
    }

    return data()[rowidx];
}


/* -------------------------------------------------------------------------- */

void qmtx_t::show(std::ostream & os, size_t width) const {
    if (data().empty())
        return;

    for (size_t rowidx = 0; rowidx < size(); ++rowidx) {
        for (size_t colidx = 0; colidx < size(); ++colidx) {
            os.width(width);
            os << data()[rowidx][colidx] << " ";
        }
        os << std::endl;
    }
}


/* -------------------------------------------------------------------------- */

void qmtx_t::_max(size_t rowidx, size_t & idx, double & max) const {
    if (rowidx >= size()) {
        assert(0);
        throw exception_t::invalid_index;
    }

    const auto & row_vector = data()[rowidx];
    idx = 0;
    max = row_vector[idx++];
    auto max_idx = idx;

    for (; idx < size(); ++idx) {
        const auto e = row_vector[idx];

        if (e > max) {
            max = e;
            max_idx = idx;
        }
    }

    idx = max_idx;
}


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

