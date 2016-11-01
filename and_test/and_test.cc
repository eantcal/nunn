/*
*  This file is part of nunnlib
*
*  nunnlib is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  nunnlib is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with nunnlib; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  US
*
*  Author: Antonino Calderone <acaldmail@gmail.com>
*
*/


/* -------------------------------------------------------------------------- */

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


/* -------------------------------------------------------------------------- */

#include "nu_perceptron.h"
#include <iostream>


/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
    try {
        nu::step_func_t step_f(0.5 /* Lo/Hi-threshold */, 0 /* Lo - Output */,
            1 /* Hi - Output */);

        nu::perceptron_t nn(2 /* inputs */, 0.2 /* learning rate */, step_f);

        // This is the bipolar-and function used for the training
        auto and_function = [](int a, int b) { return a & b; };


        // ---- TRAINING
        // ---------------------------------------------------------

        nu::perceptron_trainer_t trainer(nn,
            2000, // Max number of epochs
            0.01 // Min error
            );

        std::cout << "AND training start ( Max epochs count="
                  << trainer.get_epochs()
                  << " Minimum error=" << trainer.get_min_err() << " )"
                  << std::endl;

        size_t epoch_n = 0;

        for (auto& training_epoch : trainer) {
            bool training_completed = false;

            double err = 0.0;

            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    training_epoch.train(
                        { double(a), double(b) }, // input vector
                        { double(and_function(a, b)) }, // target

                        // cost function
                        [&err](nu::perceptron_t& net, const double& target) {
                            err = net.error(target);
                            return err;
                        });
                }
            }

            if (epoch_n++ % 100 == 0)
                std::cout << "Epoch #" << epoch_n << " Err = " << err
                          << std::endl;

            if (err < trainer.get_min_err())
                break;
        }

        // ---- TEST
        // -------------------------------------------------------------

        std::cout << " AND Test " << std::endl;

        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                double output = 0.0;
                nu::vector_t<double> input_vec{ double(a), double(b) };

                nn.set_inputs(input_vec);
                nn.feed_forward();
                output = nn.get_sharp_output();

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
    } catch (nu::perceptron_t::exception_t& e) {
        std::cerr << "nu::perceptron_t::exception_t n# " << int(e) << std::endl;

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
