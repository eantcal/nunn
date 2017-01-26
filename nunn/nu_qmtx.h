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

#ifndef __NU_QMTX_H__
#define __NU_QMTX_H__


/* -------------------------------------------------------------------------- */ 

#include <iostream>
#include "nu_vector.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

struct qmtx_t {
public:
    using vect_t = vector_t< double >;
    using data_t = std::vector< vect_t >;

    enum class exception_t {
        invalid_index
    };

    qmtx_t(const data_t& m) : _data(m) {}
    qmtx_t() = delete;
    qmtx_t(size_t size);

    void fill(const double& value) noexcept;

    const size_t size() const noexcept {
        return data().size();
    }

    friend std::ostream& operator<<(std::ostream& os, const qmtx_t& m) {
        m.show(os);
        return os;
    }

    double max(size_t rowidx) const;
    size_t maxarg(size_t rowidx) const;

    vect_t & operator[](const size_t& rowidx);
    const vect_t & operator[](const size_t& rowidx) const;

    void normalize();

protected:
    data_t & data() noexcept { 
        return _data; 
    }
    
    const data_t & data() const noexcept { 
        return _data; 
    }

    void show(std::ostream & os, size_t width = 3) const;


private:
    void _max(size_t rowidx, size_t & idx, double & max) const;

    data_t _data;
};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_QMTX_H__
