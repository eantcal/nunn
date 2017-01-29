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
 * Implementing a 3-bit counter using a neural network
 *
 */


/* -------------------------------------------------------------------------- */

#include "nu_mlpnn.h"
#include "nu_stepf.h"

#include <chrono>
#include <iostream>
#include <map>
#include <thread>


/* -------------------------------------------------------------------------- */

using neural_net_t = nu::mlp_neural_net_t;
using trainer_t = nu::mlp_nn_trainer_t;

/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
    using vect_t = neural_net_t::rvector_t;

    // Topology is a vector of positive integers
    // First one represents the input layer size
    // Last one represents the output layer size
    // All other values represent the hidden layers from input to output
    // The topology vector must be at least of 3 items and all of them must be
    // non-zero positive integer values
    neural_net_t::topology_t topology = {
        3,  // input layer takes a two dimensional vector as input
        20, // hidden layer size
        3   // output
    };

    try {

        // Construct the network using topology, learning rate and momentum
        neural_net_t nn{
            topology,
            0.05, // learning rate
            0.0,  // momentum
        };


        /*------------- Create a training set ---------------------------------
         */

        using training_set_t =
          std::map<std::vector<double>, std::vector<double>>;

        training_set_t traing_set = {
            { { 0, 0, 0 }, { 0, 0, 1 } }, { { 0, 0, 1 }, { 0, 1, 0 } },
            { { 0, 1, 0 }, { 0, 1, 1 } }, { { 0, 1, 1 }, { 1, 0, 0 } },
            { { 1, 0, 0 }, { 1, 0, 1 } }, { { 1, 0, 1 }, { 1, 1, 0 } },
            { { 1, 1, 0 }, { 1, 1, 1 } }, { { 1, 1, 1 }, { 0, 0, 0 } },
        };


        /*------------- Perform net training  ---------------------------------
         */

        const size_t EPOCHS = 40000;
        const double MIN_ERR = 0.001;

        // Create a trainer object
        trainer_t trainer(nn,
                          EPOCHS, // Max number of epochs
                          MIN_ERR // Min error
                          );

        std::cout << "Counter training start ( Max epochs count="
                  << trainer.get_epochs()
                  << " Minimum error=" << trainer.get_min_err() << " )"
                  << std::endl;

        // Called to print out training progress
        auto progress_cbk = [EPOCHS](neural_net_t& n,
                                     const nu::vector_t<double>& i,
                                     const nu::vector_t<double>& t,
                                     size_t epoch, size_t sample, double err) {
            if (epoch % 500 == 0 && sample == 0)
                std::cout << "Epoch completed "
                          << (double(epoch) / double(EPOCHS)) * 100.0
                          << "% Err=" << err * 100.0 << "%" << std::endl;
        };


        // Used by trainer to calculate the net error to
        // be compared with min error (MIN_ERR)
        auto err_cost_f = [](neural_net_t& net,
                             const neural_net_t::rvector_t& target) {
            return net.mean_squared_error(target);
        };


        // Train the net
        trainer.run_training<training_set_t>(traing_set, err_cost_f,
                                             progress_cbk);


        /*------------- Do final counter test
         * -------------------------------------
         */

        // Step function
        auto step_f =
          nu::step_func_t(0.5 /*threshold*/, 0 /* LO */, 1 /* HI */);

        std::cout << std::endl << "Counter Test " << std::endl;

        vect_t input_vec{ 0, 0, 0 };
        vect_t output_vec{ 0, 0, 0 };

        while (1) {
            nn.set_inputs(input_vec);
            nn.feed_forward();
            nn.get_outputs(output_vec);

            // Dump the network status

            std::cout << "  Input  : " << nu::vector_t<>(input_vec)
                      << std::endl;
            std::cout << "  Output : " << nu::vector_t<>(output_vec)
                      << std::endl;
            for (auto& item : output_vec) {
                item = item > 0.5 ? 1.0 : 0.0;
            }
            input_vec = output_vec;
            std::cout << "E|Output|: " << nu::vector_t<>(output_vec)
                      << std::endl;

            std::cout << "-------------------------------" << std::endl
                      << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

        std::cout << "Test completed successfully" << std::endl;
    } catch (neural_net_t::exception_t& e) {
        std::cerr << "nu::mlp_neural_net_t::exception_t n# " << int(e)
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
