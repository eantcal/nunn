//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

/*
 * Neural Network-Based 3-Bit Counter Implementation
 * -------------------------------------------------
 * This section of the code demonstrates the implementation of a 3-bit counter
 * using a neural network. The neural network is designed to emulate the behavior
 * of a digital counter that increments its binary value. Each bit in the counter
 * is represented by a neuron, and the network cycles through binary states from
 * 000 to 111, mimicking a traditional 3-bit counter. This implementation showcases
 * the adaptability of neural networks in replicating logic-based operations typically
 * performed by digital circuits.
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

    // Define the network topology using a vector of positive integers.
    // Each integer represents the number of neurons in a layer.
    // - The first element specifies the size of the input layer.
    // - The last element specifies the size of the output layer.
    // - Elements in between represent the sizes of hidden layers, in order from input to output.
    // Note:
    // - The topology vector must contain at least three elements.
    // - All elements must be non-zero positive integers.
    // Example:
    // - 3 in the first position: the input layer has 3 neurons (suitable for a 3-dimensional input vector).
    // - 20 in the second position: the hidden layer has 20 neurons.
    // - 3 in the third position: the output layer has 3 neurons.
    NeuralNet::Topology topology = { 3, 20, 3 };

    try {

        // Construct the network using topology, learning rate and momentum
        NeuralNet nn {
            topology,
            0.05, // learning rate
            0.0, // momentum
        };


        /*------------- Create a training set ---------------------------------
         */

        using TrainingSet = std::map<std::vector<double>, std::vector<double>>;

        TrainingSet traing_set = {
            { { 0, 0, 0 }, { 0, 0, 1 } },
            { { 0, 0, 1 }, { 0, 1, 0 } },
            { { 0, 1, 0 }, { 0, 1, 1 } },
            { { 0, 1, 1 }, { 1, 0, 0 } },
            { { 1, 0, 0 }, { 1, 0, 1 } },
            { { 1, 0, 1 }, { 1, 1, 0 } },
            { { 1, 1, 0 }, { 1, 1, 1 } },
            { { 1, 1, 1 }, { 0, 0, 0 } },
        };

        // ------------ Perform Neural Network Training ------------
        // This section is responsible for training the neural network.
        // It involves feeding the network with training data, adjusting weights
        // based on error rates, and iterating this process over several epochs.
        // The goal is to minimize the error rate.

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
                               [[maybe_unused]] const nu::Vector& i,
                               [[maybe_unused]] const nu::Vector& t,
                               size_t epoch,
                               size_t sample,
                               double err) {
            if (epoch % 500 == 0 && sample == 0)
                std::cout << "Epoch completed "
                          << (double(epoch) / double(EPOCHS)) * 100.0
                          << "% Err=" << err * 100.0 << "%" << std::endl;

            return false;
        };

        // Used by trainer to calculate the net error to
        // be compared with min error (MIN_ERR)
        auto errCost = [](NeuralNet& net, const NeuralNet::FpVector& target) {
            return net.calcMSE(target);
        };

        // Train the net
        trainer.runTraining<TrainingSet>(traing_set, errCost, progressCbk);

        // ------------ Final Counter Test ------------
        //   This section conducts a final test after all processing or training cycles.

        std::cout << std::endl
                  << "Counter Test " << std::endl;

        vect_t input_vec { 0, 0, 0 };
        vect_t output_vec { 0, 0, 0 };

        while (1) {
            nn.setInputVector(input_vec);
            nn.feedForward();
            nn.copyOutputVector(output_vec);

            // Dump the network status

            std::cout << "  Input  : " << nu::Vector(input_vec) << std::endl;
            std::cout << "  Output : " << nu::Vector(output_vec) << std::endl;
            for (auto& item : output_vec) {
                item = item > 0.5 ? 1.0 : 0.0;
            }
            input_vec = output_vec;
            std::cout << "E|Output|: " << nu::Vector(output_vec) << std::endl;

            std::cout << "-------------------------------" << std::endl
                      << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "Test completed successfully" << std::endl;
    } catch (const NeuralNet::SizeMismatchException& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Check for configuration parameters and retry" << std::endl;
        return 1;
    } catch (const NeuralNet::InvalidSStreamFormatException& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Check for configuration parameters and retry" << std::endl;
        return 1;
    } catch (const NeuralNet::UserdefCostfNotDefinedException& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Check for configuration parameters and retry" << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Fatal error. Check for configuration parameters and retry"
                  << std::endl;

        return 1;
    }

    return 0;
}
