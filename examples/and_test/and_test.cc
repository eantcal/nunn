//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//




/*
 * AND function implemented using a Perceptron neural net
 *
 * AND is a typical example of linearly separable function. This type
 * of function can be learned by a single Perceptron neural net
 *
 * AND takes two input arguments with values in [0,1]
 * and returns one output in [0,1], as specified in the following table:
 *
 *  x1 x2 |  y
 * ---+---+----
 *  0 | 0 |  0
 *  0 | 1 |  0
 *  1 | 0 |  0
 *  1 | 1 |  1
 *
 * It computes the logical-AND, which yields 1 if and only if the two
 * inputs have 1 values.
 *
 */




#include "nu_perceptron.h"
#include <iostream>




int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    try {
        nu::StepFunction step_f(0.5 /* Lo/Hi-threshold */, 0 /* Lo - Output */,
                               1 /* Hi - Output */);

        nu::Perceptron nn(2 /* inputs */, 0.2 /* learning rate */, step_f);

        // This is the bipolar-and function used for the training
        auto and_function = [](int a, int b) { return a & b; };


        // ---- TRAINING
        // ---------------------------------------------------------

        nu::PerceptronTrainer trainer(nn,
                                         2000, // Max number of epochs
                                         0.01  // Min error
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
                      { double(a), double(b) },       // input vector
                      { double(and_function(a, b)) }, // target

                      // cost function
                      [&err](nu::Perceptron& net, const double& target) {
                          err = net.error(target);
                          return err;
                      });
                }
            }

            if (epoch_n++ % 100 == 0)
                std::cout << "Epoch #" << epoch_n << " Err = " << err
                          << std::endl;

            if (err < trainer.getMinErr())
                break;
        }

        // ---- TEST
        // -------------------------------------------------------------

        std::cout << " AND Test " << std::endl;

        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                double output = 0.0;
                nu::Vector<double> input_vec{ double(a), double(b) };

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
    } catch (nu::Perceptron::Exception& e) {
        std::cerr << "nu::Perceptron::Exception n# " << int(e) << std::endl;

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


