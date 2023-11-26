//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include "nu_learner_listener.h"

#include <unordered_map>

namespace nu {

template<
    class Action,
    class State,
    class Agent,
    class Policy,
    class ActionRewardMap = std::unordered_map<Action, double>,
    class QMap = std::unordered_map<State, ActionRewardMap>>
class Sarsa {
public:
    using reward_t = double;
    using Listener = LearnerListener;

    Sarsa(Listener* listener = nullptr) noexcept
      : _listener(listener)
    {
    }

    Sarsa(const Sarsa&) = default;
    Sarsa& operator=(const Sarsa&) = default;

    double getLearningRate() const noexcept { return _learningRate; }

    double getDiscountRate() const noexcept { return _discountRate; }

    void setLearningRate(const double& lr) const noexcept
    {
        _learningRate = lr;
    }

    void setDiscountRate(const double& dr) const noexcept
    {
        _discountRate = dr;
    }

    Action selectAction(const Agent& agent, const Policy& policy = Policy())
    {
        return policy.template getLearnedAction<QMap>(agent, getQMap());
    }

    // learn episode
    double learn(Agent& agent, const Policy& policy = Policy())
    {

        size_t moveCnt = 0;

        Action action = policy.template selectAction<QMap>(agent, getQMap());

        auto state = agent.getCurrentState();

        double reward = 0;

        while (!agent.goal()) {
            if (_listener && !_listener->notify(reward, moveCnt++)) {
                break;
            }

            reward += updateQ(agent, policy, state, action);
            ++moveCnt;
        }

        return reward;
    }

    const QMap& getQMap() const noexcept { return _qMap; }

  protected:
    QMap& getQMap() noexcept { return _qMap; }

    double updateQ(Agent& agent,
                   const Policy& policy,
                   State& state,
                   Action& action)
    {
        auto& qsa = getQMap()[agent.getCurrentState()][action];

        // update agent state
        agent.doAction(action);

        // get current agent state
        const auto& state1 = agent.getCurrentState();

        // get a reward for current state
        const auto reward = agent.reward();

        auto action1 = policy.template selectAction<QMap>(agent, getQMap());

        qsa += getLearningRate() *
               (reward + getDiscountRate() * getQMap()[state1][action1] - qsa);

        state = state1;
        action = action1;

        return qsa;
    }

  private:
    double _learningRate = 0.1;
    double _discountRate = 0.9;

    QMap _qMap;
    Policy _policy;

    Listener* _listener = nullptr;
};

}
