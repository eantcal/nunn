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
*  Author: <antonino.calderone@ericsson.com>, <acaldmail@gmail.com>
*
*/


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
 *  x1 x2 |  y   
 * ---+---+----
 *  0 | 0 |  0
 *  0 | 1 |  1
 *  1 | 0 |  1
 *  1 | 1 |  0
 *
 * XOR computes the logical exclusive-or, which yields 1 if and 
 * only if the two inputs have different values.
 *
 * So, this classification can not be solved with linear separation, 
 * but is very easy for an MLP to generate a non-linear solution to.
 *
 */


/* -------------------------------------------------------------------------- */

#include "nu_rmlpnn.h"
#include <iostream>
#include <map>


/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
   using vect_t = nu::rmlp_neural_net_t::rvector_t;

   // Topology is a vector of positive integers
   // First one represents the input layer size
   // Last one represents the output layer size
   // All other values represent the hidden layers from input to output
   // The topology vector must be at least of 3 items and all of them must be
   // non-zero positive integer values
   nu::rmlp_neural_net_t::topology_t topology =
   {
      2, // input layer takes a two dimensional vector as input
      2, // hidden layer size
      1  // output
   };

   try {

      // Construct the network using given topology and 
      // learning rate and momentum 
      nu::rmlp_neural_net_t nn
      {
         topology,
         0.4, // learing rate
         0.9, // momentum
      };


      // This is the bipolar-xor function used for the training
      auto xor = [](int a, int b) { return a ^ b; };

      nu::rmlp_nn_trainer_t trainer(
         nn, 
         20000,  // Max number of epochs
         0.01   // Min error 
      );

      std::cout
         << "XOR training start ( Max epochs count=" << trainer.get_epochs()
         << " Minimum error=" << trainer.get_min_err() << " )"
         << std::endl;

      // Create a training set
      using training_set_t = std::map< std::vector<double>, std::vector<double> >;
      training_set_t traing_set;

      for ( int a = 0; a < 2; ++a )
      {
         for ( int b = 0; b < 2; ++b )
         {
            const std::vector<double> v{ double(a), double(b) };
            const std::vector<double> t{ double(xor(a, b)) };
            traing_set.insert(std::make_pair(v, t));
         }
      }

      // Train the net
      trainer.train<training_set_t>(
         traing_set, 
         [](
            nu::rmlp_neural_net_t& net,
            const nu::rmlp_neural_net_t::rvector_t & target) -> double
            { 
               static size_t i = 0;

               if (i++ % 200 == 0 )
                  std::cout << ">";

               return net.mean_squared_error(target); 
            }
      );

      // Perform final XOR test
      auto step_f = [](double x) { return x < 0.5 ? 0 : 1; };

      std::cout << " XOR Test " << std::endl;

      for ( int a = 0; a < 2; ++a )
      {
         for ( int b = 0; b < 2; ++b )
         {
            vect_t output_vec{ 0.0 };
            vect_t input_vec{ double(a), double(b) };

            nn.set_inputs(input_vec);
            nn.feed_forward();
            nn.get_outputs(output_vec);

            // Dump the network status
            std::cout << nn;

            std::cout << "-------------------------------" << std::endl;

            auto net_res = step_f(output_vec[0]);

            std::cout
               << a << " xor " << b << " = " << net_res << std::endl;

            auto xor_res = xor(a, b);

            // In case you play with configuration parameters 
            // and break the code :-)

            if ( xor_res != net_res )
            {
               std::cerr
                  << "ERROR!: xor(" << a << "," << b << ") !="
                  << xor_res
                  << std::endl;

               return 1;
            }

            std::cout << "-------------------------------" << std::endl;
         }
      }

      std::cout << "Test completed successfully" << std::endl;
   }
   catch ( nu::rmlp_neural_net_t::exception_t & e )
   {
      std::cerr 
         << "nu::mlp_neural_net_t::exception_t n# " << int(e) << std::endl;
      
      std::cerr
         << "Check for configuration parameters and retry" << std::endl;

      return 1;
   }
   catch ( ... )
   {
      std::cerr
         << "Fatal error. Check for configuration parameters and retry" 
         << std::endl;

      return 1;
   }

   return 0;
}

/* -------------------------------------------------------------------------- */

