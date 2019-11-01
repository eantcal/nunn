//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/*
  This is an implementation of a Perceptron Neural Network which learns by
  example.

  You can give it examples of what you want the network to do and the algorithm
  changes the network's weights. When training is finished, the net will give
  you
  the required output for a particular input.
*/


/* -------------------------------------------------------------------------- */

#ifndef __NU_PERCEPTRON_H__
#define __NU_PERCEPTRON_H__


/* -------------------------------------------------------------------------- */

#include "nu_neuron.h"

#include "nu_stepf.h"
#include "nu_trainer.h"
#include "nu_vector.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

//! This class represents a Perceptron neural net
struct Perceptron {
    using FpVector = Vector<double>;

    enum class Exception {
        size_mismatch,
        invalid_sstream_format
    };

    //! default ctor
    Perceptron() = default;

    //! ctor
    Perceptron(const size_t& inputSize, double learningRate = 0.1,
                 StepFunction step_f = StepFunction());

    //! Create a perceptron using data serialized into the given stream
    Perceptron(std::stringstream& ss) { load(ss); }

    //! copy-ctor
    Perceptron(const Perceptron& nn) = default;

    //! move-ctor
    Perceptron(Perceptron&& nn)
      : _inputSize(std::move(nn._inputSize))
      , _learningRate(std::move(nn._learningRate))
      , _inputVector(std::move(nn._inputVector))
      , _neuron(std::move(nn._neuron))
    {
    }

    //! default assignment operator
    Perceptron& operator=(const Perceptron& nn) = default;

    //! default assignment-move operator
    Perceptron& operator=(Perceptron&& nn) {
        if (this != &nn) {
            _inputSize = std::move(nn._inputSize);
            _learningRate = std::move(nn._learningRate);
            _inputVector = std::move(nn._inputVector);
            _neuron = std::move(nn._neuron);
        }

        return *this;
    }

    //! Return the number of inputs
    size_t getInputSize() const noexcept { 
        return _inputVector.size(); 
    }

    //! Return current learning rate
    double getLearningRate() const noexcept { 
        return _learningRate; 
    }

    //! Change net learning rate
    void setLearningRate(double new_rate) { 
        _learningRate = new_rate; 
    }

    //! Set net inputs
    void setInputVector(const FpVector& inputs) {
        if (inputs.size() != _inputVector.size())
            throw Exception::size_mismatch;

        _inputVector = inputs;
    }

    //! Get net inputs
    void getInputVector(FpVector& inputs) const noexcept { 
        inputs = _inputVector; 
    }

    //! Get net output
    double getOutput() const noexcept { 
        return _neuron.output; 
    }

    //! Return f(getOutput()), where f is the step function
    double getSharpOutput() const noexcept {
        return _step_f(getOutput());
    }

    //! Fire all neurons of the net and calculate the outputs
    void feedForward() noexcept;

    //! Fire the neuron, calculate the output
    //! then apply the learning algorithm to the net
    void backPropagate(const double& target, double& output) noexcept;

    //! Fire the neuron, calculate the output
    //! then apply the learning algorithm to the net
    void backPropagate(const double& target) noexcept {
        double output;
        backPropagate(target, output);
    }

    //! Compute global error
    double error(const double& target) const noexcept {
        return std::abs(target - getOutput());
    }

    //! Build the net by using data of the given string stream
    std::stringstream& load(std::stringstream& ss);

    //! Save net status into the given string stream
    std::stringstream& save(std::stringstream& ss) noexcept;

    //! Print the net state out to the given ostream
    std::ostream& dump(std::ostream& os) noexcept;

    //! Build the net by using data of the given string stream
    friend 
    std::stringstream& operator>>(std::stringstream& ss, Perceptron& net) {
        return net.load(ss);
    }

    //! Save net status into the given string stream
    friend 
    std::stringstream& operator<<(std::stringstream& ss, Perceptron& net) noexcept {
        return net.save(ss);
    }

    //! Print the net state out to the given ostream
    friend
    std::ostream& operator<<(std::ostream& os, Perceptron& net) noexcept {
        return net.dump(os);
    }

    //! Reset all net weights using new random values
    void reshuffleWeights() noexcept;

private:
    constexpr static const char* ID_ANN = "perceptron";
    constexpr static const char* ID_NEURON = "neuron";
    constexpr static const char* ID_INPUTS = "inputs";

    StepFunction _step_f;
    size_t _inputSize;
    double _learningRate = 0.1;
    FpVector _inputVector;
    Neuron _neuron;
};


/* -------------------------------------------------------------------------- */

//! The perceptron trainer class is a helper class for training perceptrons
struct PerceptronTrainer : public NNTrainer<Perceptron, nu::Vector<double>, double>
{
    PerceptronTrainer(Perceptron& nn, size_t epochs, double minErr)
      : NNTrainer<Perceptron, nu::Vector<double>, double>(nn, epochs, minErr)
    {
    }
};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_PERCEPTRON_H__
