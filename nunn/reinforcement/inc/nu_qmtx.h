//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_QMTX_H__
#define __NU_QMTX_H__


/* -------------------------------------------------------------------------- */ 

#include <iostream>
#include "nu_vector.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

struct QMatrix {
public:
    using vect_t = Vector< double >;
    using data_t = std::vector< vect_t >;

    enum class Exception {
        invalid_index
    };

    QMatrix(const data_t& m) : _data(m) {}
    QMatrix() = delete;
    QMatrix(size_t size);
    QMatrix(const QMatrix& other) : _data(other._data) {}

    QMatrix& operator=(const QMatrix& other) {
        if (this != &other) {
            _data = other._data;
        }
        return *this;
    }

    void fill(const double& value) noexcept;

    const size_t size() const noexcept {
        return data().size();
    }

    friend std::ostream& operator<<(std::ostream& os, const QMatrix& m) {
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
