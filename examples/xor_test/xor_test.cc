//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

/*
 * Solving the XOR Problem with nunn Lib
 *
 * A typical example of non-linearly separable function is the XOR.
 * Implementing the XOR function is a classic problem in neural networks.
 *
 * This function takes two input arguments with values in [0,1]
 * and returns one output in [0,1], as specified in the following table:
 *
 *  x1| x2|  y
 * ---+---+----
 *  0 | 0 |  0
 *  0 | 1 |  1
 *  1 | 0 |  1
 *  1 | 1 |  0
 *
 * XOR computes the logical exclusive-or, which yields 1 if and
 * only if the two inputs x1 and x2 have different values.
 * This classification can not be solved with linear separation,
 * but is very easy for an MLP to generate a non-linear solution to.
 *
 */


/* -------------------------------------------------------------------------- */

#include "nu_mlpnn.h"
#include "nu_stepf.h"

#include <iostream>
#include <map>


/* -------------------------------------------------------------------------- */

using neural_net_t = nu::MlpNN;
using trainer_t = nu::MlpNNTrainer;

/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
    using vect_t = neural_net_t::FpVector;

    // Topology is a vector of positive integers
    // First one represents the input layer size
    // Last one represents the output layer size
    // All other values represent the hidden layers from input to output
    // The topology vector must be at least of 3 items and all of them must be
    // non-zero positive integer values
    neural_net_t::Topology topology = {
        2, // input layer takes a two dimensional vector as input
        2, // hidden layer size
        1  // output
    };

    try {

        // Construct the network using topology, learning rate and momentum
        neural_net_t nn{
            topology,
            0.4, // learning rate
            0.9, // momentum
        };


        /*------------- Create a training set ---------------------------------
         */

        using training_set_t =
          std::map<std::vector<double>, std::vector<double>>;

        training_set_t traing_set = { { { 0, 0 }, { 0 } },
                                      { { 0, 1 }, { 1 } },
                                      { { 1, 0 }, { 1 } },
                                      { { 1, 1 }, { 0 } } };


        /*------------- Perform net training  ---------------------------------
         */

        const size_t EPOCHS = 40000;
        const double MIN_ERR = 0.01;

        // Create a trainer object
        trainer_t trainer(nn,
                          EPOCHS, // Max number of epochs
                          MIN_ERR // Min error
                          );

        std::cout << "XOR training start ( Max epochs count="
                  << trainer.getEpochs()
                  << " Minimum error=" << trainer.getMinErr() << " )"
                  << std::endl;

        // Called to print out training progress
        auto progressCbk = [EPOCHS](neural_net_t& n,
                                     const nu::Vector<double>& i,
                                     const nu::Vector<double>& t,
                                     size_t epoch, size_t sample, double err) {
            if (epoch % 400 == 0 && sample == 0)
                std::cout << "Epoch completed "
                          << (double(epoch) / double(EPOCHS)) * 100.0
                          << "% Err=" << err * 100.0 << "%" << std::endl;

            return false;
        };


        // Used by trainer to calculate the net error to
        // be compared with min error (MIN_ERR)
        auto errCost = [](neural_net_t& net,
                             const neural_net_t::FpVector& target) {
            return net.calcMSE(target);
        };


        // Train the net
        trainer.runTraining<training_set_t>(traing_set, errCost,
                                             progressCbk);


        /*------------- Do final XOR test -------------------------------------
         */

        // Step function
        auto step_f =
          nu::StepFunction(0.5 /*threshold*/, 0 /* LO */, 1 /* HI */);

        std::cout << std::endl << "XOR Test " << std::endl;

        for (const auto& sample : traing_set) {
            vect_t output_vec{ 0.0 };

            nn.setInputVector(sample.first);
            nn.feedForward();
            nn.copyOutputVector(output_vec);

            // Dump the network status
            std::cout << nn;

            std::cout << "-------------------------------" << std::endl;

            auto net_res = step_f(output_vec[0]);

            std::cout << sample.first[0] << " xor " << sample.first[1] << " = "
                      << net_res << std::endl;

            auto xor_res = sample.second[0];

            // In case you play with configuration parameters
            // and break the code :-)

            if (xor_res != net_res) {
                std::cerr << "ERROR!: xor(" << sample.first[0] << ","
                          << sample.first[1] << ") !=" << xor_res << std::endl;

                return 1;
            }

            std::cout << "-------------------------------" << std::endl
                      << std::endl;
        }

        std::cout << "Test completed successfully" << std::endl;
    } catch (neural_net_t::Exception& e) {
        std::cerr << "nu::MlpNN::Exception n# " << int(e)
                  << std::endl;

        std::cerr << "Check for configuration parameters and retry"
                  << std::endl;

        return 1;
    } catch (...) {
        std::cerr << "Fatal error. Check for configuration parameters and retry"
                  << std::endl;

        return 1;
    }

    return 0;
}

/* -------------------------------------------------------------------------- */
