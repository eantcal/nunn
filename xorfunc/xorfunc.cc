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

#include "nu_mlpnn.h"
#include <iostream>


/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
   // Topology is a vector of positive integers
   // First one represents the input layer size
   // Last one represents the output layer size
   // All other values represent the hidden layers from input to output
   // The topology vector must be at least of 3 items and all of them must be
   // non-zero positive integer values
   nu::mlp_neural_net_t::topology_t topology =
   {
      2, // input layer takes a two dimensional vector as input
      2, // hidden layer size
      1  // output
   };

   try {

      // Construct the network using given topology and 
      // learning rate and momentum 
      nu::mlp_neural_net_t nn
      {
         topology,
         0.4, // learing rate
         0.9, // momentum
      };

      using vect_t = nu::mlp_neural_net_t::rvector_t;

      // This is the bipolar-xor function used for the training
      auto xor = [](int a, int b) { return a ^ b; };

      int epochs = 2000;     // Max number of epochs
      double min_err = 0.01; // Min err

      std::cout
         << "XOR training start (Max epochs count=" << epochs
         << " Minimum performance gradient=" << min_err << " )"
         << std::endl;

      double err = 1.0;
      int current_epoch = 0;

      //Repeat until err is less than min_err threshold or
      //epochs iterations have been completed
      while ( epochs-- )
      {
         err = 0.0;

         for ( int a = 0; a < 2; ++a )
         {
            for ( int b = 0; b < 2; ++b )
            {
               int target = xor(a, b);

               // Output is represented by one-dimensional vector
               vect_t target_vec{ double(target) };

               // Input is represented by two-dimensional vector
               nn.set_inputs({ double(a), double(b) });

               // Call back-propagation algo
               nn.back_propagate(target_vec);

               // Compute the mean squared error for this sample
               err += nn.mean_squared_error(target_vec);
            }
         }

         // Compute the error of a complete truth table sample
         err /= 4.0;

         // Periodically, show the training progress
         if ( epochs % 100 == 0 )
            std::cout
            << "Epoch #" << current_epoch
            << " Err = " << err << std::endl;
         ++current_epoch;

         // Terminate the loop if err is less than min_err threshold
         if ( err < min_err )
            break;
      }


      // Perform final XOR test
      //
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
                  << xor(a, b)
                  << std::endl;

               return 1;
            }

            std::cout << "-------------------------------" << std::endl;
         }
      }
   }
   catch ( nu::mlp_neural_net_t::exception_t & e )
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

