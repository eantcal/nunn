//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

/**
 * @file nu_perceptron.h
 * @brief Implementation of a Perceptron Neural Network.
 *
 * The Perceptron is a type of artificial neuron which uses a step function for
 * activation. It is a fundamental building block for certain types of neural
 * networks and serves as a binary classifier in its simplest form. This
 * implementation allows for training the perceptron using examples to adjust
 * its weights, thereby learning to produce the desired output for a given
 * input.
 */

#pragma once

#include "nu_neuron.h"
#include "nu_stepf.h"
#include "nu_trainer.h"
#include "nu_vector.h"

#include <string_view>

namespace nu {

/**
 * @class Perceptron
 * @brief Represents a Perceptron neural network.
 *
 * This class models a perceptron, a type of single-layer neural network.
 * It provides functionalities for setting the learning rate, input vector, and
 * performing operations like feedforward and backpropagation.
 */
struct Perceptron
{
    using FpVector = Vector<double>;

    enum class Exception
    {
        size_mismatch,
        invalid_sstream_format
    };

    //! Default constructor.
    Perceptron() = default;

    /**
     * @brief Constructs a perceptron with a specified input size.
     * @param inputSize The size of the input vector.
     * @param learningRate The learning rate of the perceptron.
     * @param step_f The step function used for neuron activation.
     */
    Perceptron(size_t inputSize,
               double learningRate = 0.1,
               StepFunction step_f = StepFunction());

    //! Constructs a perceptron from serialized data in a stream.
    Perceptron(std::stringstream& ss) { load(ss); }

    //! Default copy and move constructors.
    Perceptron(const Perceptron& nn) = default;
    Perceptron(Perceptron&& nn) = default;

    //! Default copy and move assignment operators.
    Perceptron& operator=(const Perceptron& nn) = default;
    Perceptron& operator=(Perceptron&& nn) = default;

    //! Returns the number of inputs to the perceptron.
    size_t getInputSize() const noexcept { return _inputVector.size(); }

    //! Returns the current learning rate of the perceptron.
    double getLearningRate() const noexcept { return _learningRate; }

    //! Sets a new learning rate for the perceptron.
    void setLearningRate(double new_rate) { _learningRate = new_rate; }

    //! Sets the input vector for the perceptron.
    void setInputVector(const FpVector& inputs);

    //! Retrieves the input vector of the perceptron.
    void getInputVector(FpVector& inputs) const noexcept
    {
        inputs = _inputVector;
    }

    //! Returns the output of the perceptron.
    double getOutput() const noexcept { return _neuron.output; }

    //! Returns the sharp output (after applying the step function) of the
    //! perceptron.
    double getSharpOutput() const noexcept { return _step_f(getOutput()); }

    //! Performs the feedforward operation.
    void feedForward() noexcept;

    //! Performs backpropagation to adjust the weights based on the target
    //! output.
    void backPropagate(const double& target, double& output) noexcept;

    //! Performs backpropagation using only the target output.
    void backPropagate(const double& target) noexcept;

    //! Computes the error between the target and actual output.
    double error(const double& target) const noexcept;

    //! Loads the perceptron configuration from a stream.
    std::stringstream& load(std::stringstream& ss);

    //! Saves the perceptron configuration to a stream.
    std::stringstream& save(std::stringstream& ss) noexcept;

    //! Dumps the state of the perceptron to an output stream.
    std::ostream& dump(std::ostream& os) noexcept;

    //! Resets the perceptron's weights to random values.
    void reshuffleWeights() noexcept;

    //! Build the net by using data of the given string stream
    friend std::stringstream& operator>>(std::stringstream& ss, Perceptron& net)
    {
        return net.load(ss);
    }

    //! Save net status into the given string stream
    friend std::stringstream& operator<<(std::stringstream& ss,
                                         Perceptron& net) noexcept
    {
        return net.save(ss);
    }

    //! Print the net state out to the given ostream
    friend std::ostream& operator<<(std::ostream& os, Perceptron& net) noexcept
    {
        return net.dump(os);
    }

  private:
    constexpr static std::string_view ID_ANN = "perceptron";
    constexpr static std::string_view ID_NEURON = "neuron";
    constexpr static std::string_view ID_INPUTS = "inputs";

    StepFunction _step_f;
    double _learningRate = 0.1;
    FpVector _inputVector;
    Neuron _neuron;
};

/**
 * @class PerceptronTrainer
 * @brief Helper class for training Perceptron neural networks.
 *
 * Facilitates the training of a perceptron by providing methods to run training
 * sessions over multiple epochs and evaluate the perceptron's performance.
 */
struct PerceptronTrainer : public NNTrainer<Perceptron, Vector<double>, double>
{
    PerceptronTrainer(Perceptron& nn, size_t epochs, double minErr)
      : NNTrainer<Perceptron, Vector<double>, double>(nn, epochs, minErr)
    {
    }
};

} // namespace nu
