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
class perceptron_t
{
  public:
    using rvector_t = vector_t<double>;

    enum class exception_t
    {
        size_mismatch,
        invalid_sstream_format
    };

    //! default ctor
    perceptron_t() = default;

    //! ctor
    perceptron_t(const size_t& n_of_inputs, double learning_rate = 0.1,
                 step_func_t step_f = step_func_t());


    //! Create a perceptron using data serialized into the given stream
    perceptron_t(std::stringstream& ss) { load(ss); }

    //! copy-ctor
    perceptron_t(const perceptron_t& nn) = default;


    //! move-ctor
    perceptron_t(perceptron_t&& nn)
      : _inputs_count(std::move(nn._inputs_count))
      , _learning_rate(std::move(nn._learning_rate))
      , _inputs(std::move(nn._inputs))
      , _neuron(std::move(nn._neuron))
    {
    }


    //! default assignment operator
    perceptron_t& operator=(const perceptron_t& nn) = default;


    //! default assignment-move operator
    perceptron_t& operator=(perceptron_t&& nn)
    {
        if (this != &nn) {
            _inputs_count = std::move(nn._inputs_count);
            _learning_rate = std::move(nn._learning_rate);
            _inputs = std::move(nn._inputs);
            _neuron = std::move(nn._neuron);
        }

        return *this;
    }


    //! Return the number of inputs
    size_t get_inputs_count() const noexcept { return _inputs.size(); }


    //! Return current learning rate
    double get_learning_rate() const noexcept { return _learning_rate; }


    //! Change net learning rate
    void set_learning_rate(double new_rate) { _learning_rate = new_rate; }


    //! Set net inputs
    void set_inputs(const rvector_t& inputs)
    {
        if (inputs.size() != _inputs.size())
            throw exception_t::size_mismatch;

        _inputs = inputs;
    }


    //! Get net inputs
    void get_inputs(rvector_t& inputs) const noexcept { inputs = _inputs; }


    //! Get net output
    double get_output() const noexcept { return _neuron.output; }


    //! Return f(get_output()), where f is the step function
    double get_sharp_output() const noexcept
    {
        return _step_f(get_output());
    }


    //! Fire all neurons of the net and calculate the outputs
    void feed_forward() noexcept;


    //! Fire the neuron, calculate the output
    //! then apply the learning algorithm to the net
    void back_propagate(const double& target, double& output) noexcept;


    //! Fire the neuron, calculate the output
    //! then apply the learning algorithm to the net
    void back_propagate(const double& target) noexcept
    {
        double output;
        back_propagate(target, output);
    }


    //! Compute global error
    double error(const double& target) const noexcept
    {
        return std::abs(target - get_output());
    }


    //! Build the net by using data of the given string stream
    std::stringstream& load(std::stringstream& ss);


    //! Save net status into the given string stream
    std::stringstream& save(std::stringstream& ss) noexcept;


    //! Print the net state out to the given ostream
    std::ostream& dump(std::ostream& os) noexcept;


    //! Build the net by using data of the given string stream
    friend std::stringstream& operator>>(std::stringstream& ss,
                                         perceptron_t& net)
    {
        return net.load(ss);
    }


    //! Save net status into the given string stream
    friend std::stringstream& operator<<(std::stringstream& ss,
                                         perceptron_t& net) noexcept
    {
        return net.save(ss);
    }


    //! Print the net state out to the given ostream
    friend std::ostream& operator<<(std::ostream& os,
                                    perceptron_t& net) noexcept
    {
        return net.dump(os);
    }


    //! Reset all net weights using new random values
    void reshuffle_weights() noexcept;

  private:
    void _back_propagate(const double_t& target,
                         const double_t& output) noexcept;

    static const char* ID_ANN;
    static const char* ID_NEURON;
    static const char* ID_INPUTS;

    step_func_t _step_f;
    size_t _inputs_count;
    double _learning_rate = 0.1;
    rvector_t _inputs;
    neuron_t<double> _neuron;
};


/* -------------------------------------------------------------------------- */

//! The perceptron trainer class is a helper class for training perceptrons
class perceptron_trainer_t
  : public nn_trainer_t<perceptron_t, nu::vector_t<double>, double>
{
  public:
    perceptron_trainer_t(perceptron_t& nn, size_t epochs, double min_err)
      : nn_trainer_t<perceptron_t, nu::vector_t<double>, double>(nn, epochs,
                                                                 min_err)
    {
    }
};


/* -------------------------------------------------------------------------- */

} // namespace nu


/* -------------------------------------------------------------------------- */

#endif // __NU_PERCEPTRON_H__
