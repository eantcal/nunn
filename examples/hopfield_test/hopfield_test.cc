//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//



/*
 * Hopfield neural network test
 * The Hopfield network is used to solve the recall problem of matching
 * copy for an input pattern to an associated pre-learned pattern
 */



#include "nu_hopfieldnn.h"
#include <iostream>



const size_t pattern_size = 100;
const size_t n_of_patterns = 5;

std::string g_learning_patterns[] = { // 0123456789
    { "   ***    "                    // 0
      "  ****    "                    // 1
      " *****    "                    // 2
      "   ***    "                    // 3
      "   ***    "                    // 4
      "   ***    "                    // 5
      "   ***    "                    // 6
      "   ***    "                    // 7
      " *******  "                    // 8
      " *******  " },                 // 9

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

std::string g_test_patterns[] = { // 0123456789
    { "   ***    "                // 0
      "   ***    "                // 1
      "   ***    "                // 2
      "   ***    "                // 3
      "   ***    "                // 4
      "   ***    "                // 5
      "   ***    "                // 6
      "   ***    "                // 7
      "   ***    "                // 8
      "   ***    " },             // 9

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



static void print_pattern(const std::string& pattern)
{
    std::cout << "+----------+" << std::endl;
    for (int y = 0; y < 10; ++y) {
        std::cout << '|';
        for (int x = 0; x < 10; ++x)
            std::cout << pattern[y * 10 + x];
        std::cout << '|';
        std::cout << std::endl;
    }
    std::cout << "+----------+" << std::endl;
    std::cout << std::endl;
}



static void print_pattern(const nu::HopfiledNN::FpVector& pattern)
{
    std::string s_pattern;
    for (int y = 0; y < 10; ++y)
        for (int x = 0; x < 10; ++x)
            s_pattern += (pattern[y * 10 + x] == 1.0 ? '*' : ' ');

    print_pattern(s_pattern);
}



static void convert_pattern_into_input_v(const std::string& pattern,
                                         nu::HopfiledNN::FpVector& input_vector)
{
    size_t i = 0;
    for (auto c : pattern) {
        input_vector[i++] = c == '*' ? 1 : -1;
    }
}



int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    nu::HopfiledNN net(pattern_size);

    std::cout << "LEARNING THE FOLLOWING IMAGES:" << std::endl;

    for (size_t i = 0; i < n_of_patterns; ++i) {
        nu::HopfiledNN::FpVector input_v(pattern_size);
        convert_pattern_into_input_v(g_learning_patterns[i], input_v);
        net.addPattern(input_v);
        print_pattern(g_learning_patterns[i]);
    }

    // Test the net
    for (size_t i = 0; i < n_of_patterns; ++i) {
        nu::HopfiledNN::FpVector input_v(pattern_size);
        nu::HopfiledNN::FpVector output_pattern(pattern_size);

        convert_pattern_into_input_v(g_test_patterns[i], input_v);
        net.recall(input_v, output_pattern);

        std::cout << std::endl
                  << std::endl
                  << std::endl
                  << " THIS IMAGE" << std::endl;

        print_pattern(g_test_patterns[i]);

        std::cout << std::endl << "  RECALLS" << std::endl;

        print_pattern(output_pattern);
    }

    return 0;
}


