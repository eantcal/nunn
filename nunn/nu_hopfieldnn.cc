//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#include "nu_hopfieldnn.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

void hopfieldnn_t::add_pattern(const FpVector& input_pattern)
{
    const auto size = getInputSize();

    if (size != input_pattern.size())
        throw Exception::size_mismatch;

    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j) {
            if (i != j)
                _w[i * size + j] += input_pattern[i] * input_pattern[j];
        }

    ++_pattern_size;
}


/* -------------------------------------------------------------------------- */

void hopfieldnn_t::recall(const FpVector& input_pattern,
                          FpVector& output_pattern)
{
    if (getInputSize() != input_pattern.size())
        throw Exception::size_mismatch;

    _s = input_pattern;
    _propagate();

    output_pattern = _s;
}


/* -------------------------------------------------------------------------- */

//! default assignment-move operator
hopfieldnn_t& hopfieldnn_t::operator=(hopfieldnn_t&& nn) noexcept
{
    if (this != &nn) {
        _s = std::move(nn._s);
        _w = std::move(nn._w);
        _pattern_size = std::move(_pattern_size);
    }

    return *this;
}


/* -------------------------------------------------------------------------- */

void hopfieldnn_t::_propagate() noexcept
{
    size_t it = 0;
    size_t last_it = 0;

    do {
        ++it;
        size_t rnd_idx = rand() % getInputSize();

        if (_propagate_neuron(rnd_idx))
            last_it = it;

    } while (it - last_it < 10 * getInputSize());
}


/* -------------------------------------------------------------------------- */

bool hopfieldnn_t::_propagate_neuron(size_t i) noexcept
{
    bool changed = false;
    double sum = 0;

    const auto size = getInputSize();

    for (size_t j = 0; j < size; ++j)
        sum += _w[i * size + j] * _s[j];

    double state = 0.0;

    if (sum != 0.0) {
        if (sum < 0.0)
            state = -1;

        if (sum > 0.0)
            state = 1;

        if (state != _s[i]) {
            changed = true;
            _s[i] = state;
        }
    }

    return changed;
}


/* -------------------------------------------------------------------------- */

std::stringstream& hopfieldnn_t::load(std::stringstream& ss)
{
    std::string s;
    ss >> s;
    if (s != hopfieldnn_t::ID_ANN)
        throw Exception::invalid_sstream_format;

    ss >> _pattern_size;


    ss >> s;
    if (s != hopfieldnn_t::ID_NEURON_ST)
        throw Exception::invalid_sstream_format;

    ss >> _s;

    ss >> s;
    if (s != hopfieldnn_t::ID_WEIGHTS)
        throw Exception::invalid_sstream_format;

    ss >> _w;

    return ss;
}


/* -------------------------------------------------------------------------- */

std::stringstream& hopfieldnn_t::save(std::stringstream& ss) noexcept
{
    ss.clear();

    ss << hopfieldnn_t::ID_ANN << std::endl;

    ss << _pattern_size << std::endl;

    ss << hopfieldnn_t::ID_NEURON_ST << std::endl;
    ss << _s << std::endl;

    ss << hopfieldnn_t::ID_WEIGHTS << std::endl;
    ss << _w << std::endl;

    return ss;
}


/* -------------------------------------------------------------------------- */

std::ostream& hopfieldnn_t::dump(std::ostream& os) noexcept
{
    os << "Hopfield " << std::endl;

    os << "\t# of patterns  " << _pattern_size << std::endl;
    os << "\tNeurons Status " << _s << std::endl;
    os << "\tNet Weights    " << _w << std::endl;

    os << std::endl;

    return os;
}


/* -------------------------------------------------------------------------- */

const char* hopfieldnn_t::ID_ANN = "hopfield";
const char* hopfieldnn_t::ID_NEURON_ST = "neuron_st";
const char* hopfieldnn_t::ID_WEIGHTS = "net_weights";


/* -------------------------------------------------------------------------- */

} // namespace nu
