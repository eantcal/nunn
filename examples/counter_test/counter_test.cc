//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//




/*
 * Implementing a 3-bit counter using a neural network
 *
 */




#include "nu_mlpnn.h"
#include "nu_stepf.h"

#include <chrono>
#include <iostream>
#include <map>
#include <thread>




using NeuralNet = nu::MlpNN;
using Trainer = nu::MlpNNTrainer;



int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    using vect_t = NeuralNet::FpVector;

    // Topology is a vector of positive integers
    // First one represents the input layer size
    // Last one represents the output layer size
    // All other values represent the hidden layers from input to output
    // The topology vector must be at least of 3 items and all of them must be
    // non-zero positive integer values
    NeuralNet::Topology topology = {
        3,  // input layer takes a two dimensional vector as input
        20, // hidden layer size
        3   // output
    };

    try {

        // Construct the network using topology, learning rate and momentum
        NeuralNet nn{
            topology,
            0.05, // learning rate
            0.0,  // momentum
        };


        /*------------- Create a training set ---------------------------------
         */

        using TrainingSet =
          std::map<std::vector<double>, std::vector<double>>;

        TrainingSet traing_set = {
            { { 0, 0, 0 }, { 0, 0, 1 } }, { { 0, 0, 1 }, { 0, 1, 0 } },
            { { 0, 1, 0 }, { 0, 1, 1 } }, { { 0, 1, 1 }, { 1, 0, 0 } },
            { { 1, 0, 0 }, { 1, 0, 1 } }, { { 1, 0, 1 }, { 1, 1, 0 } },
            { { 1, 1, 0 }, { 1, 1, 1 } }, { { 1, 1, 1 }, { 0, 0, 0 } },
        };


        /*------------- Perform net training  ---------------------------------
         */

        constexpr size_t EPOCHS = 40000;
        constexpr double MIN_ERR = 0.001;

        // Create a trainer object
        Trainer trainer(nn,
                          EPOCHS, // Max number of epochs
                          MIN_ERR // Min error
                          );

        std::cout << "Counter training start ( Max epochs count="
                  << trainer.getEpochs()
                  << " Minimum error=" << trainer.getMinErr() << " )"
                  << std::endl;

        // Called to print out training progress
        auto progressCbk = []([[maybe_unused]] NeuralNet& n,
                                     [[maybe_unused]] const nu::Vector<double>& i,
                                     [[maybe_unused]] const nu::Vector<double>& t,
                                     size_t epoch, size_t sample, double err) {
            if (epoch % 500 == 0 && sample == 0)
                std::cout << "Epoch completed "
                          << (double(epoch) / double(EPOCHS)) * 100.0
                          << "% Err=" << err * 100.0 << "%" << std::endl;

            return false;
        };


        // Used by trainer to calculate the net error to
        // be compared with min error (MIN_ERR)
        auto errCost = [](NeuralNet& net,
                             const NeuralNet::FpVector& target) {
            return net.calcMSE(target);
        };


        // Train the net
        trainer.runTraining<TrainingSet>(traing_set, errCost,
                                             progressCbk);


        /*------------- Do final counter test
         * -------------------------------------
         */

        // Step function
        auto step_f =
          nu::StepFunction(0.5 /*threshold*/, 0 /* LO */, 1 /* HI */);

        std::cout << std::endl << "Counter Test " << std::endl;

        vect_t input_vec{ 0, 0, 0 };
        vect_t output_vec{ 0, 0, 0 };

        while (1) {
            nn.setInputVector(input_vec);
            nn.feedForward();
            nn.copyOutputVector(output_vec);

            // Dump the network status

            std::cout << "  Input  : " << nu::Vector<>(input_vec)
                      << std::endl;
            std::cout << "  Output : " << nu::Vector<>(output_vec)
                      << std::endl;
            for (auto& item : output_vec) {
                item = item > 0.5 ? 1.0 : 0.0;
            }
            input_vec = output_vec;
            std::cout << "E|Output|: " << nu::Vector<>(output_vec)
                      << std::endl;

            std::cout << "-------------------------------" << std::endl
                      << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "Test completed successfully" << std::endl;
    } catch (NeuralNet::Exception& e) {
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


