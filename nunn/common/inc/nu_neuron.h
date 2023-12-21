//
// This file is part of the nunn Library
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

//! Represents a single neuron within a neural network layer.
struct Neuron {
    //! Vector of synaptic weights. Each weight corresponds to the connection strength with an input.
    Vector weights;

    //! Adjustment vector for weights used during the training process (backpropagation).
    Vector deltaW;

    //! Bias term for the neuron, contributing to the net input signal beyond weighted inputs.
    double bias { .0 };

    //! Output value of the neuron after applying the activation function.
    double output { .0 };

    //! Error gradient value for the neuron, used in training to update weights and bias.
    double error { .0 };

    //! Serializes the neuron's state (bias, weights, delta weights) into a stringstream for saving.
    friend std::stringstream& operator<<(std::stringstream& ss, const Neuron& n) noexcept
    {
        ss << n.bias << std::endl;
        ss << n.weights << std::endl;
        ss << n.deltaW << std::endl;

        return ss;
    }

    //! Loads the neuron's state (bias, weights, delta weights) from a stringstream.
    friend std::stringstream& operator>>(std::stringstream& ss, Neuron& n) noexcept
    {
        ss >> n.bias;
        ss >> n.weights;
        ss >> n.deltaW;

        return ss;
    }

    //! Outputs the neuron's state in JSON format, useful for data interchange or human-readable saves.
    std::ostream& toJson(std::ostream& ss) noexcept
    {
        ss << "{\"bias\":" << bias << ",";
        ss << "\"weights\":";
        weights.toJson(ss) << ",";
        ss << "\"deltaW\":";
        deltaW.toJson(ss) << "}";

        return ss;
    }

    //! Resizes the weight and delta weight vectors to a new specified size. Used when initializing or modifying the neuron.
    void resize(size_t size) noexcept
    {
        weights.resize(size);
        deltaW.resize(size);
    }
};

}
