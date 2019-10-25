//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_E_GREEDY_POLICY_H__
#define __NU_E_GREEDY_POLICY_H__


/* -------------------------------------------------------------------------- */

#include "nu_random_gen.h"
#include <cmath>
#include <cassert>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template<class Action, class Agent, class RndGen = RandomGenerator<>>
class EGreedyPolicy {
public:
    void setEpsilon(const double & e) noexcept {
        _epsilon = e;
    }

    double getEpsilon() const noexcept {
        return _epsilon;
    }

    template<class QMap>
    Action selectAction(
        const Agent& agent, 
        QMap & qMap, 
        bool dontExplore = false) const
    {
        auto validActions = agent.getValidActions();

        assert(!validActions.empty());

        Action action = validActions[0];

        double reward = 0;

        if (dontExplore || _rndGen() > getEpsilon()) {
            // get current agent state
            const auto & agentState = agent.getCurrentState();

            reward = qMap[agentState][action];

            for (const auto & anAction : validActions) {
                const auto & val = qMap[agentState][anAction];
                if (val > reward) {
                    reward = val;
                    action = anAction;
                }
            }
        }

        if (reward == .0) {
            action =
                validActions[size_t(_rndGen() * double(validActions.size()))];
        }

        return action;
    }

    template<class QMap>
    Action getLearnedAction(const Agent& agent, QMap & qMap) const {
        return selectAction(agent, qMap, true);
    }

private:
    double _epsilon = .1;
    mutable RndGen _rndGen;

};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_E_GREEDY_POLICY_H__

