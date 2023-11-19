//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


#include "nu_hopfieldnn.h"

namespace nu {

void HopfiledNN::addPattern(const FpVector& input_pattern)
{
    const auto size = getInputSize();

    if (size != input_pattern.size())
        throw Exception::size_mismatch;

    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j) {
            if (i != j)
                _w[i * size + j] += input_pattern[i] * input_pattern[j];
        }

    ++_patternSize;
}

void HopfiledNN::recall(const FpVector& input_pattern,
                          FpVector& output_pattern)
{
    if (getInputSize() != input_pattern.size())
        throw Exception::size_mismatch;

    _s = input_pattern;
    _propagate();

    output_pattern = _s;
}

void HopfiledNN::_propagate() noexcept
{
    size_t it = 0;
    size_t last_it = 0;

    do {
        ++it;
        size_t rnd_idx = 
            size_t ( getInputSize() * _rndgen() ) % getInputSize();

        if (_propagateNeuron(rnd_idx))
            last_it = it;

    } while (it - last_it < 10 * getInputSize());
}

bool HopfiledNN::_propagateNeuron(size_t i) noexcept
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

std::stringstream& HopfiledNN::load(std::stringstream& ss)
{
    std::string s;
    ss >> s;
    if (s != HopfiledNN::ID_ANN)
        throw Exception::invalid_sstream_format;

    ss >> _patternSize;


    ss >> s;
    if (s != HopfiledNN::ID_NEURON_ST)
        throw Exception::invalid_sstream_format;

    ss >> _s;

    ss >> s;
    if (s != HopfiledNN::ID_WEIGHTS)
        throw Exception::invalid_sstream_format;

    ss >> _w;

    return ss;
}

std::stringstream& HopfiledNN::save(std::stringstream& ss) noexcept
{
    ss.clear();

    ss << HopfiledNN::ID_ANN << std::endl;

    ss << _patternSize << std::endl;

    ss << HopfiledNN::ID_NEURON_ST << std::endl;
    ss << _s << std::endl;

    ss << HopfiledNN::ID_WEIGHTS << std::endl;
    ss << _w << std::endl;

    return ss;
}

std::ostream& HopfiledNN::dump(std::ostream& os) noexcept
{
    os << "Hopfield " << std::endl;

    os << "\t# of patterns  " << _patternSize << std::endl;
    os << "\tNeurons Status " << _s << std::endl;
    os << "\tNet Weights    " << _w << std::endl;

    os << std::endl;

    return os;
}

const char* HopfiledNN::ID_ANN = "hopfield";
const char* HopfiledNN::ID_NEURON_ST = "neuron_st";
const char* HopfiledNN::ID_WEIGHTS = "net_weights";

} // namespace nu
