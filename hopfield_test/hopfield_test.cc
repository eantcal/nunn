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
 * Hopfield neural network test
 * The Hopfield network is used to solve the recall problem of matching 
 * copy for an input pattern to an associated pre-learned pattern
 */


/* -------------------------------------------------------------------------- */

#include "nu_hopfieldnn.h"
#include <iostream>


/* -------------------------------------------------------------------------- */

const size_t pattern_size = 100;
const size_t n_of_patterns = 5;

std::string g_learning_patterns[] = 
{  // 0123456789
   { "   ***    "   //0
     "  ****    "   //1
     " *****    "   //2
     "   ***    "   //3
     "   ***    "   //4
     "   ***    "   //5
     "   ***    "   //6
     "   ***    "   //7
     " *******  "   //8
     " *******  " },//9

   { "**********"
     "**********"
     "**********"
     "**********"
     "**********"
     "          "
     "          "
     "          "
     "          "
     "          " },

   { "*****     "
     "*****     "
     "*****     "
     "*****     "
     "*****     "
     "     *****"
     "     *****"
     "     *****"
     "     *****"
     "     *****" },

   { "**********"
     "**********"
     "**      **"
     "**      **"
     "**      **"
     "**********"
     "**********"
     "**      **"
     "**      **"
     "**      **" },

   { "**********"
     "*        *"
     "* ****** *"
     "* *    * *"
     "* * ** * *"
     "* * ** * *"
     "* *    * *"
     "* ****** *"
     "*        *"
     "**********" } 
};


std::string g_test_patterns[] =
{  // 0123456789
 { "   ***    "   //0
   "   ***    "   //1
   "   ***    "   //2
   "   ***    "   //3
   "   ***    "   //4
   "   ***    "   //5
   "   ***    "   //6
   "   ***    "   //7
   "   ***    "   //8
   "   ***    " },//9

 { "**********"
   "**********"
   "          "
   "          "
   "          "
   "          "
   "          "
   "          "
   "          "
   "          " },

 { "          "
   "          "
   "*****     "
   "*****     "
   "*****     "
   "     *****"
   "     *****"
   "     *****"
   "          "
   "          " },

 { "**********"
   "*        *"
   "*        *"
   "*        *"
   "*        *"
   "**********"
   "**********"
   "*        *"
   "*        *"
   "*        *" },

 { "**********"
   "*        *"
   "* ****** *"
   "* *    * *"
   "* *    * *"
   "* *    * *"
   "* *    * *"
   "* ****** *"
   "*        *"
   "**********" }
};


/* -------------------------------------------------------------------------- */

static void print_pattern(const std::string& pattern)
{
   std::cout << "+----------+" << std::endl;
   for (int y = 0; y < 10; ++y)
   {
      std::cout << '|';
      for (int x = 0; x < 10; ++x)
         std::cout << pattern[y * 10 + x];
      std::cout << '|';
      std::cout << std::endl;
   }
   std::cout << "+----------+" << std::endl;
   std::cout << std::endl;
}


/* -------------------------------------------------------------------------- */

static void print_pattern(const nu::hopfieldnn_t::rvector_t& pattern)
{
   std::string s_pattern;
   for (int y = 0; y < 10; ++y)
      for (int x = 0; x < 10; ++x)
         s_pattern += (pattern[y * 10 + x] == 1.0 ? '*' : ' ');

   print_pattern(s_pattern);
}


/* -------------------------------------------------------------------------- */

static void convert_pattern_into_input_v(
   const std::string& pattern,
   nu::hopfieldnn_t::rvector_t & input_vector)
{
   size_t i = 0;
   for (auto c : pattern)
      input_vector[i++] = c == '*' ? 1 : -1;
}



/* -------------------------------------------------------------------------- */

int main(int argc, char* argv[])
{
   nu::hopfieldnn_t net(pattern_size);

   std::cout << "LEARNING THE FOLLOWING IMAGES:" << std::endl;

   for (int i = 0; i < n_of_patterns; ++i)
   {
      nu::hopfieldnn_t::rvector_t input_v(pattern_size);
      convert_pattern_into_input_v(g_learning_patterns[i], input_v);
      net.add_pattern(input_v);
      print_pattern(g_learning_patterns[i]);
   }

   // Test the net
   for (int i = 0; i < n_of_patterns; ++i)
   {
      nu::hopfieldnn_t::rvector_t input_v(pattern_size);
      nu::hopfieldnn_t::rvector_t output_pattern(pattern_size);

      convert_pattern_into_input_v(g_test_patterns[i], input_v);
      net.recall(input_v, output_pattern);

      std::cout << std::endl << std::endl;
      std::cout << std::endl << " THIS IMAGE" << std::endl;

      print_pattern(g_test_patterns[i]);

      std::cout << std::endl << "  RECALLS" << std::endl;

      print_pattern(output_pattern);
   }

   return 0;
}

/* -------------------------------------------------------------------------- */

