//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include "nu_random_gen.h"
#include <cassert>
#include <cmath>
#include <limits>

namespace nu {

template <class Action, class Agent, class RndGen = RandomGenerator<>>
class SoftmaxPolicy {
public:
    void setTemperature(const double& temperature) noexcept
    {
        _temperature = temperature;
    }

    double getTemperature() const noexcept { return _temperature; }

    template <class QMap>
    Action selectAction(const Agent& agent, QMap& qMap) const
    {
        // Get agent to reward map
        auto actionReward = qMap[agent.getCurrentState()];
        auto validActions = agent.getValidActions();

        assert(!validActions.empty());

        // Guard against a zero/negative temperature, which would otherwise
        // divide by zero and yield exp(+/-inf).
        const double temperature = getTemperature() > 0.0
            ? getTemperature()
            : std::numeric_limits<double>::min();

        decltype(actionReward) quasiProbs;
        double sumReward = 0;

        for (const auto& item : validActions) {
            const auto numerator = std::exp(actionReward[item] / temperature);
            quasiProbs[item] = numerator;
            sumReward += numerator;
        }

        assert(sumReward != 0);

        // Roulette-wheel selection. Iterate the valid actions (deterministic
        // order) and accumulate normalized probabilities until we pass the
        // cutoff. If floating-point rounding leaves the running sum just below
        // the cutoff, fall back to the last valid action instead of running
        // past the end of the container.
        const auto cutoff = _rndGen();

        double sum = 0;
        Action selected = validActions.back();

        for (const auto& item : validActions) {
            sum += quasiProbs[item] / sumReward;
            if (sum > cutoff) {
                selected = item;
                break;
            }
        }

        return selected;
    }

    template <class QMap>
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

        // TODO(v3.0): the exact == 0.0 test conflates "best Q value is zero"
        // with "state never explored". Revisit together with a proper
        // exploration schedule when the policy layer is redesigned.
        if (reward == .0) {
            action = selectAction(agent, qMap);
        }

        return action;
    }

private:
    double _temperature = 1.0;
    mutable RndGen _rndGen;
};

}
