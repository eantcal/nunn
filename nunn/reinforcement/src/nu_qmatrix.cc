//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_qmatrix.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <vector>

namespace nu {

QMatrix::QMatrix(size_t n_of_states)
{
    _data.resize(n_of_states);
    for (auto& row : _data) {
        row.resize(n_of_states);
    }
}

void QMatrix::fill(const double& value) noexcept
{
    for (auto& row : data()) {
        std::fill(row.begin(), row.end(), value);
    }
}

double QMatrix::max(size_t rowidx) const
{
    size_t maxidx = 0;
    double maxvalue = 0;
    _max(rowidx, maxidx, maxvalue);
    return maxvalue;
}

size_t QMatrix::maxarg(size_t rowidx) const
{
    size_t maxidx = 0;
    double maxvalue = 0;
    _max(rowidx, maxidx, maxvalue);
    return maxidx;
}

void QMatrix::normalize()
{
    double globalMax = std::numeric_limits<double>::lowest();

    for (const auto& row : _data) {
        auto rowMax = *std::max_element(row.cbegin(), row.cend());
        globalMax = std::max(globalMax, rowMax);
    }

    if (globalMax != 0) {
        double scaleFactor = 100.0 / globalMax;
        for (auto& row : _data) {
            std::transform(row.begin(), row.end(), row.begin(),
                [scaleFactor](double val) { return val * scaleFactor; });
        }
    }
}


QMatrix::vect_t& QMatrix::operator[](const size_t& rowidx)
{
    if (rowidx >= size()) {
        assert(0);
        throw InvalidIndexException();
    }

    return data()[rowidx];
}

const QMatrix::vect_t& QMatrix::operator[](const size_t& rowidx) const
{
    if (rowidx >= size()) {
        assert(0);
        throw InvalidIndexException();
    }

    return data()[rowidx];
}

void QMatrix::show(std::ostream& os, size_t width) const
{
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

void QMatrix::_max(size_t rowidx, size_t& maxIdx, double& maxValue) const
{
    if (rowidx >= size()) {
        throw InvalidIndexException();
    }

    const auto& row = _data[rowidx];
    auto maxElementIter = std::max_element(row.cbegin(), row.cend());

    maxValue = *maxElementIter;
    maxIdx = static_cast<size_t>(std::distance(row.cbegin(), maxElementIter));
}

}
