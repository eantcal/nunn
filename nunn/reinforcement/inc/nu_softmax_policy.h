//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_SOFTMAX_POLICY_H__
#define __NU_SOFTMAX_POLICY_H__


/* -------------------------------------------------------------------------- */

#include "nu_random_gen.h"
#include <cassert>
#include <cmath>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template<class Action, class Agent, class RndGen = RandomGenerator<>>
class SoftmaxPolicy
{
  public:
    void setTemperature(const double& temperature) noexcept
    {
        _temperature = temperature;
    }

    double getTemperature() const noexcept { return _temperature; }

    template<class QMap>
    Action selectAction(const Agent& agent, QMap& qMap) const
    {
        // Get agent to reward map
        auto actionReward = qMap[agent.getCurrentState()];
        auto validActions = agent.getValidActions();
        decltype(actionReward) quasiProbs;

        double sumReward = 0;

        for (const auto& item : validActions) {
            const auto reward = actionReward[item];
            const auto numerator = std::exp(reward / getTemperature());
            quasiProbs[item] = numerator;
            sumReward += numerator;
        }

        assert(sumReward != 0);

        // Select an action
        const auto cutoff = _rndGen();

        double sum = 0;
        auto it = quasiProbs.begin();

        for (; it != quasiProbs.end(); ++it) {

            const auto prob = it->second / sumReward;
            sum += prob;

            if (sum > cutoff) {
                return it->first;
            }
        }

        assert(0);

        return it->first;
    }

    template<class QMap>
    Action getLearnedAction(const Agent& agent, QMap& qMap) const
    {
        auto validActions = agent.getValidActions();

        assert(!validActions.empty());

        Action action = validActions[0];

        double reward = 0;

        const auto& agentState = agent.getCurrentState();

        reward = qMap[agentState][action];

        for (const auto& anAction : validActions) {
            const auto& val = qMap[agentState][anAction];
            if (val > reward) {
                reward = val;
                action = anAction;
            }
        }

        if (reward == .0) {
            action = selectAction(agent, qMap);
        }

        return action;
    }

  private:
    double _temperature = 1.0;
    mutable RandomGenerator<> _rndGen;
};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_SOFTMAX_POLICY_H__
