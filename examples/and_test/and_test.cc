//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

/*
 * Perceptron Neural Network Implementation of the AND Function
 * ------------------------------------------------------------
 * This code implements the logical AND function using a Perceptron,
 * a type of single-layer neural network. The AND function is a classic
 * example of a linearly separable function, making it well-suited for
 * learning with a Perceptron.
 *
 * The AND function accepts two binary inputs (0 or 1) and produces a single
 * binary output based on the following truth table:
 *
 *  Input1 (x1) | Input2 (x2) | Output (y)
 *  --------------------------------------
 *       0      |      0      |     0
 *       0      |      1      |     0
 *       1      |      0      |     0
 *       1      |      1      |     1
 *
 * The output is 1 only when both inputs are 1, embodying the logical AND operation.
 * This neural network model learns to replicate this behavior by adjusting its
 * weights based on the provided training data.
 */

#include "nu_perceptron.h"
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    try {
        nu::StepFunction step_f(
            0.5 /* Lo/Hi-threshold */, 0 /* Lo - Output */, 1 /* Hi - Output */);

        nu::Perceptron nn(2 /* inputs */, 0.2 /* learning rate */, step_f);

        // This is the bipolar-and function used for the training
        auto and_function = [](int a, int b) { return a & b; };

        // ---- TRAINING
        // ---------------------------------------------------------

        nu::PerceptronTrainer trainer(nn,
            2000, // Max number of epochs
            0.01 // Min error
        );

        std::cout << "AND training start ( Max epochs count="
                  << trainer.getEpochs()
                  << " Minimum error=" << trainer.getMinErr() << " )"
                  << std::endl;

        size_t epoch_n = 0;

        for (auto& training_epoch : trainer) {
            double err = 0.0;

            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    training_epoch.train(
                        { double(a), double(b) }, // input vector
                        { double(and_function(a, b)) }, // target

                        // cost function
                        [&err](nu::Perceptron& net, const double& target) {
                            err = net.error(target);
                            return err;
                        });
                }
            }

            if (epoch_n++ % 100 == 0) {
                std::cout << "Epoch #" << epoch_n << " Err = " << err
                          << std::endl;
            }

            if (err < trainer.getMinErr()) {
                break;
            }
        }

        // ---- TEST
        // -------------------------------------------------------------

        std::cout << " AND Test " << std::endl;

        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                double output = 0.0;
                nu::Vector input_vec { double(a), double(b) };

                nn.setInputVector(input_vec);
                nn.feedForward();
                output = nn.getSharpOutput();

                // Dump the network status
                std::cout << nn;

                std::cout << "-------------------------------" << std::endl;

                std::cout << a << " and " << b << " = " << output << std::endl;

                auto and_res = and_function(a, b);

                // In case you'd play with configuration parameters
                // and break the code :-)
                if (int(and_res) != int(output)) {
                    std::cerr << "ERROR!: and(" << a << "," << b
                              << ") !=" << and_res << std::endl;

                    return 1;
                }

                std::cout << "-------------------------------" << std::endl;
            }
        }

        std::cout << "Test completed successfully" << std::endl;
    } catch (const nu::Perceptron::SizeMismatchException& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << "Check for configuration parameters and retry" << std::endl;
        return 1;
    } catch (const nu::Perceptron::InvalidSStreamFormatException& e) {
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
