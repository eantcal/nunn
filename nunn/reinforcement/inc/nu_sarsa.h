// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.

#pragma once

#include "nu_learner_listener.h"
#include <unordered_map>

namespace nu {

// Sarsa is a template class implementing the SARSA (State-Action-Reward-State-Action) learning algorithm.
// It is a form of reinforcement learning that uses a policy-based approach to find a policy that maximizes
// the cumulative reward of an agent.
template <class Action, class State, class Agent, class Policy,
    class ActionRewardMap = std::unordered_map<Action, double>,
    class QMap = std::unordered_map<State, ActionRewardMap>>
class Sarsa {
public:
    using reward_t = double;
    using Listener = LearnerListener;

    // Constructor, optionally accepting a listener for tracking learning events.
    explicit Sarsa(Listener* listener = nullptr) noexcept
        : _listener(listener)
    {
    }

    Sarsa(const Sarsa&) = default;
    Sarsa& operator=(const Sarsa&) = default;

    // Returns the learning rate, a value influencing the rate of training.
    double getLearningRate() const noexcept { return _learningRate; }

    // Returns the discount rate, a value that reduces the future rewards.
    double getDiscountRate() const noexcept { return _discountRate; }

    // Sets the learning rate.
    void setLearningRate(const double& lr) const noexcept { _learningRate = lr; }

    // Sets the discount rate.
    void setDiscountRate(const double& dr) const noexcept { _discountRate = dr; }

    // Selects an action based on the given policy.
    Action selectAction(const Agent& agent, const Policy& policy = Policy())
    {
        return policy.template getLearnedAction<QMap>(agent, getQMap());
    }

    // Learns an episode. Returns the total reward accumulated during the episode.
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

    // Returns the current Q-map, which holds the state-action values.
    const QMap& getQMap() const noexcept { return _qMap; }

protected:
    // Internal method to access and modify the Q-map.
    QMap& getQMap() noexcept { return _qMap; }

    // Updates the Q-value for the given state and action.
    double updateQ(Agent& agent, const Policy& policy, State& state, Action& action)
    {
        auto& qsa = getQMap()[agent.getCurrentState()][action];
        agent.doAction(action);
        const auto& state1 = agent.getCurrentState();
        const auto reward = agent.reward();
        auto action1 = policy.template selectAction<QMap>(agent, getQMap());

        qsa += getLearningRate() * (reward + getDiscountRate() * getQMap()[state1][action1] - qsa);
        state = state1;
        action = action1;

        return qsa;
    }

private:
    double _learningRate { 0.1 }; // Default learning rate
    double _discountRate { 0.9 }; // Default discount rate

    QMap _qMap; // Map storing the state-action values
    Policy _policy; // Policy used in SARSA learning

    Listener* _listener = nullptr; // Optional listener for learning events
};

}
