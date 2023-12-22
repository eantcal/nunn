//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#include "nu_hopfieldnn.h"
#include <algorithm>
#include <random>
#include <span>

namespace nu {

void HopfieldNN::clear() noexcept
{
    std::fill(_s.begin(), _s.end(), 0.0); // Reset neuron states to all zeros
    std::fill(_w.begin(), _w.end(), 0.0); // Reset weights to all zeros
    _patternSize = 0;
}

void HopfieldNN::addPattern(const FpVector& input_pattern)
{
    const auto size = getInputSize();

    if (size != input_pattern.size()) {
        throw SizeMismatchException();
    }

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            if (i != j)
                _w[i * size + j] += input_pattern[i] * input_pattern[j];
        }
    }

    ++_patternSize;
}

void HopfieldNN::recall(const FpVector& input_pattern, FpVector& output_pattern)
{
    if (getInputSize() != input_pattern.size()) {
        throw SizeMismatchException();
    }

    _s = input_pattern;
    _propagate();

    output_pattern = _s;
}

void HopfieldNN::_propagate() noexcept
{
    const size_t size = getInputSize();
    std::uniform_int_distribution<size_t> dist(0, size - 1);

    size_t it = 0, last_it = 0;

    do {
        ++it;
        size_t rnd_idx = dist(_rndgen); // Generate a random index

        if (_propagateNeuron(rnd_idx)) {
            last_it = it;
        }

    } while (it - last_it < 10 * size);
}

bool HopfieldNN::_propagateNeuron(size_t i) noexcept
{
    double sum = std::inner_product(_w.begin() + i * getInputSize(),
        _w.begin() + (i + 1) * getInputSize(),
        _s.begin(),
        0.0);
    double state = (sum > 0.0) - (sum < 0.0);

    if (state != _s[i]) {
        _s[i] = state;
        return true;
    }

    return false;
}

std::stringstream& HopfieldNN::load(std::stringstream& ss)
{
    std::string s;
    ss >> s;
    if (s != ID_ANN) {
        throw InvalidSStreamFormatException();
    }

    ss >> _patternSize;

    ss >> s;
    if (s != ID_NEURON_ST) {
        throw InvalidSStreamFormatException();
    }

    ss >> _s;

    ss >> s;
    if (s != ID_WEIGHTS) {
        throw InvalidSStreamFormatException();
    }

    ss >> _w;

    return ss;
}

std::stringstream& HopfieldNN::save(std::stringstream& ss) noexcept
{
    ss.clear();

    ss << ID_ANN << "\n"
       << _patternSize << "\n"
       << ID_NEURON_ST << "\n"
       << _s << "\n"
       << ID_WEIGHTS << "\n"
       << _w;

    return ss;
}

std::ostream& HopfieldNN::dump(std::ostream& os) noexcept
{
    os << "Hopfield Neural Network\n# of Patterns: " << _patternSize
       << "\nNeuron States: " << _s << "\nNet Weights: " << _w << "\n";

    return os;
}

} // namespace nu
