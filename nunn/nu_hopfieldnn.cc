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

#include "nu_hopfieldnn.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

void hopfieldnn_t::add_pattern(const rvector_t& input_pattern)
{
    const auto size = get_inputs_count();

    if (size != input_pattern.size())
        throw exception_t::size_mismatch;

    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j) {
            if (i != j)
                _w[i * size + j] += input_pattern[i] * input_pattern[j];
        }

    ++_pattern_size;
}


/* -------------------------------------------------------------------------- */

void hopfieldnn_t::recall(
    const rvector_t& input_pattern, rvector_t& output_pattern)
{
    if (get_inputs_count() != input_pattern.size())
        throw exception_t::size_mismatch;

    _s = input_pattern;
    _propagate();

    output_pattern = _s;
}


/* -------------------------------------------------------------------------- */

//! default assignment-move operator
hopfieldnn_t& hopfieldnn_t::operator=(hopfieldnn_t&& nn) NU_NOEXCEPT
{
    if (this != &nn) {
        _s = std::move(nn._s);
        _w = std::move(nn._w);
        _pattern_size = std::move(_pattern_size);
    }

    return *this;
}


/* -------------------------------------------------------------------------- */

void hopfieldnn_t::_propagate() NU_NOEXCEPT
{
    size_t it = 0;
    size_t last_it = 0;

    do {
        ++it;
        size_t rnd_idx = rand() % get_inputs_count();

        if (_propagate_neuron(rnd_idx))
            last_it = it;

    } while (it - last_it < 10 * get_inputs_count());
}


/* -------------------------------------------------------------------------- */

bool hopfieldnn_t::_propagate_neuron(size_t i) NU_NOEXCEPT
{
    bool changed = false;
    double sum = 0;

    const auto size = get_inputs_count();

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
        throw exception_t::invalid_sstream_format;

    ss >> _pattern_size;


    ss >> s;
    if (s != hopfieldnn_t::ID_NEURON_ST)
        throw exception_t::invalid_sstream_format;

    ss >> _s;

    ss >> s;
    if (s != hopfieldnn_t::ID_WEIGHTS)
        throw exception_t::invalid_sstream_format;

    ss >> _w;

    return ss;
}


/* -------------------------------------------------------------------------- */

std::stringstream& hopfieldnn_t::save(std::stringstream& ss) NU_NOEXCEPT
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

std::ostream& hopfieldnn_t::dump(std::ostream& os) NU_NOEXCEPT
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
