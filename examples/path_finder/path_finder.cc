//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

/*
   Shortest path finder example.
*/



#include "nu_qlgraph.h"
#include <iostream>




const size_t NumberOfStates = 6;
const size_t NumberOfEpisodies = 1000;
const size_t GoalState = 5;




int main()
{
    /*              ____________________
                    |                   \
                    v                   |
                   [1]----------\       |
                    ^           |       |
                    |           |       |
                    v           v       |
       [2]<------->[3]    /--->[5] ---->/   [5] is Goal state
                    ^     \___/ ^ \
                    |           |  \
                    v           |  |
       [0]<------->[4]__________/  |
                    ^              |
                    |              |
                    \______________/

    */

    std::cout << "Path finder example (using Q-Larning)" << std::endl
              << std::endl
              << "                ____________________  " << std::endl
              << "                |                   \\  " << std::endl
              << "                v                   |  " << std::endl
              << "               [1]----------\\       |  " << std::endl
              << "                ^           |       |  " << std::endl
              << "                |           |       |  " << std::endl
              << "                v           v       |  " << std::endl
              << "   [2]<------->[3]    /--->[5] ---->/  " << std::endl
              << "                ^     \\___/ ^ \\" << std::endl
              << "                |           |  \\" << std::endl
              << "                v           |  |" << std::endl
              << "   [0]<------->[4]__________/  |" << std::endl
              << "                ^              |" << std::endl
              << "                |              |" << std::endl
              << "                \\______________/" << std::endl
              << std::endl
              << std::endl
              << "Goal is state [" << GoalState << "]" << std::endl;


    nu::QLGraph ql(NumberOfStates,
                   GoalState,

                   // graph topology
                   { // from  you can go to ...
                     { 0, { 4 } },
                     { 1, { 3, 5 } },
                     { 2, { 3 } },
                     { 3, { 1, 2, 4 } },
                     { 4, { 0, 3, 5 } },
                     { 5, { 1, 4, 5 } } });

    ql.learn(NumberOfEpisodies);

#ifdef _DEBUG
    auto& q = ql.get_q_mtx();
    std::cout << "Q=" << std::endl << q << std::endl;
#endif

    std::cout << std::endl
              << "From | Shortest path to " << GoalState << std::endl;
    std::cout << "---- | ------------------- " << std::endl;

    for (size_t init_state = 0; init_state < NumberOfStates; ++init_state) {

        size_t _curState = init_state;

        bool goal = false;

        std::cout << "  " << _curState << "  |  ";

        while (!goal) {

            auto nextState = ql.getNextStateFor(_curState);

            _curState = nextState;
            goal = _curState == GoalState;

            std::cout << _curState << "  ";
        }

        std::cout << std::endl;
    }


    return 0;
}
