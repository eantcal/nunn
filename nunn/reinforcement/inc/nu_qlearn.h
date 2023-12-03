//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include "nu_learner_listener.h"

#include <unordered_map>
#include <memory>

namespace nu {

template <class Action, // possible action in state S
    class State, // representation of the environmental states
    class Agent, // the Q-learning agent
    class Policy, // policy to follow
    class ActionRewardMap = std::unordered_map<Action, double>,
    class QMap = std::unordered_map<State, ActionRewardMap>>
class QLearn {
public:
    using reward_t = double; // the reward
    using Listener = LearnerListener;

    QLearn(std::shared_ptr<Listener> listener = nullptr) noexcept
        : _listener(listener)
    {
    }

    QLearn(const QLearn&) = default;
    QLearn& operator=(const QLearn&) = default;

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
        double reward = 0;

        while (!agent.goal()) {
            if (auto listener=_listener.lock(); listener && !listener->notify(reward, moveCnt++)) {
                break;
            }

            reward += updateQ(agent, policy);
        }

        return reward;
    }

    const QMap& getQMap() const noexcept { return _qMap; }

protected:
    QMap& getQMap() noexcept { return _qMap; }

    double updateQ(Agent& agent, const Policy& policy)
    {
        Action action = policy.template selectAction<QMap>(agent, getQMap());

        auto& qsa = getQMap()[agent.getCurrentState()][action];

        // update agent state
        agent.doAction(action);

        // get current agent state
        const auto& agentState = agent.getCurrentState();

        // get a reward for current state
        const auto reward = agent.reward();

        // get a list of valid actions for current state
        const auto validActions = agent.getValidActions();

        auto max = getQMap()[agentState][validActions[0]];

        for (const auto& anAction : validActions) {
            auto val = getQMap()[agentState][anAction];
            if (val > max) {
                max = val;
            }
        }

        qsa += getLearningRate() * (reward + getDiscountRate() * max - qsa);

        return qsa;
    }

private:
    double _learningRate { 0.1 };
    double _discountRate { 0.9 };

    QMap _qMap;
    Policy _policy;

    std::weak_ptr<Listener> _listener;
};

}
