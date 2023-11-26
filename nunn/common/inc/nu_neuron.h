//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


#pragma once

#include "nu_vector.h"
#include <iostream>
#include <sstream>

namespace nu {

//! This class represents a neuron of a neural net neuron layer
struct Neuron {
    //! neuron weights
    Vector<double> weights;

    //! amount by which weights will change
    Vector<double> deltaW;

    //! neuron bias
    double bias = 0.0;

    //! neuron output
    double output = 0.0;

    //! error field used by learning algorithm
    double error = 0.0;

    //! Save neuron status into a given string stream
    friend 
    std::stringstream& operator<<(std::stringstream& ss, const Neuron& n) noexcept {
        ss << n.bias << std::endl;
        ss << n.weights << std::endl;
        ss << n.deltaW << std::endl;

        return ss;
    }

    //! Load neuron status from a given string stream
    friend 
    std::stringstream& operator>>(std::stringstream& ss, Neuron& n) noexcept {
        ss >> n.bias;
        ss >> n.weights;
        ss >> n.deltaW;

        return ss;
    }

    //! Save JSON formatted neuron status into a given string stream
    std::ostream& formatJson(std::ostream& ss) noexcept {
        ss << "{\"bias\":"<< bias << ",";
        ss << "\"weights\":";
        weights.formatJson(ss) << ",";
        ss << "\"deltaW\":";
        deltaW.formatJson(ss) << "}";

        return ss;
    }

    //! Resize both the weights and deltaW vectors
    void resize(size_t size) noexcept {
        weights.resize(size);
        deltaW.resize(size);
    }
};

}

