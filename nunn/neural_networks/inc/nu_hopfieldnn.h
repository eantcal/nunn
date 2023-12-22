//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

/**
 * @file hopfield_nn.h
 * @brief Implementation of the Hopfield Neural Network.
 *
 * The Hopfield Neural Network is a form of recurrent artificial neural network
 * popularized by John Hopfield. Hopfield networks serve as content-addressable
 * memory systems with binary threshold nodes. They are guaranteed to converge
 * to a local minimum, but convergence to one of the stored patterns is not
 * guaranteed.
 */

#pragma once

#include "nu_stepf.h"
#include "nu_vector.h"

#include <list>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string_view>


namespace nu {

/**
 * @class HopfieldNN
 * @brief Implementation of a Hopfield Neural Network.
 *
 * This class represents a Hopfield Neural Network, which is a type of recurrent
 * neural network. It can store patterns and recall them when provided with a
 * key pattern. The network's capacity and the number of stored patterns can be
 * queried.
 */
class HopfieldNN {
public:
    using FpVector = Vector;

    class SizeMismatchException : public std::runtime_error {
    public:
        SizeMismatchException()
            : std::runtime_error("Size mismatch")
        {
        }
    };

    class InvalidSStreamFormatException : public std::runtime_error {
    public:
        InvalidSStreamFormatException()
            : std::runtime_error("Invalid stringstream format")
        {
        }
    };

    //! Constructs a Hopfield Neural Network.
    explicit HopfieldNN(size_t inputSize = 0) noexcept
        : _s(inputSize)
        , _w(inputSize * inputSize)
    {
        std::random_device rd;
        _rndgen.seed(rd());
    }

    HopfieldNN() = delete;

    /**
     * @brief Returns the maximum number of patterns the network can store.
     * @return size_t The maximum number of storable patterns.
     */
    size_t getCapacity() const noexcept
    {
        return static_cast<size_t>(0.138 * static_cast<double>(_s.size()));
    }

    /**
     * @brief Gets the count of patterns added to the network.
     * @return size_t The number of stored patterns.
     */
    size_t getPatternsCount() const noexcept { return _patternSize; }

    //! Adds a specified pattern to the network.
    void addPattern(const FpVector& input_pattern);

    //! Attempts to recall a pattern using a given input pattern.
    void recall(const FpVector& input_pattern, FpVector& output_pattern);

    //! Loads network configuration from a stringstream.
    HopfieldNN(std::stringstream& ss) { load(ss); }

    //! Default copy and move constructors.
    HopfieldNN(const HopfieldNN& nn) = default;
    HopfieldNN(HopfieldNN&& nn) noexcept = default;

    //! Default copy and move assignment operators.
    HopfieldNN& operator=(const HopfieldNN& nn) = default;
    HopfieldNN& operator=(HopfieldNN&& nn) noexcept = default;

    //! Gets the number of inputs to the network.
    size_t getInputSize() const noexcept { return _s.size(); }

    //! Loads network configuration from a given string stream.
    std::stringstream& load(std::stringstream& ss);

    //! Saves network configuration to a given string stream.
    std::stringstream& save(std::stringstream& ss) noexcept;

    //! Dumps the network state to an output stream.
    std::ostream& dump(std::ostream& os) noexcept;

    //! Clears the network state, resetting it to its initial configuration.
    void clear() noexcept;

private:
    static constexpr std::string_view ID_ANN = "HopfieldNN";
    static constexpr std::string_view ID_WEIGHTS = "Weights";
    static constexpr std::string_view ID_NEURON_ST = "NeuronStates";

    StepFunction step_f = StepFunction(0, -1, 1);

    void _propagate() noexcept;
    bool _propagateNeuron(size_t i) noexcept;

    FpVector _s; // Neuron states
    FpVector _w; // Weights matrix
    size_t _patternSize = 0;

    std::mt19937 _rndgen;
};

// Serialization operators
inline std::stringstream& operator>>(std::stringstream& ss, HopfieldNN& net)
{
    return net.load(ss);
}

inline std::stringstream& operator<<(std::stringstream& ss,
    HopfieldNN& net) noexcept
{
    return net.save(ss);
}

inline std::ostream& operator<<(std::ostream& os, HopfieldNN& net) noexcept
{
    return net.dump(os);
}

} // namespace nu
