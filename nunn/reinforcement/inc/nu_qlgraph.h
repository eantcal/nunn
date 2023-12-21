//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include "nu_qmatrix.h"
#include "nu_random_gen.h"

#include <cassert>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nu {

// QLGraph implements a Q-Learning graph for reinforcement learning.
// It uses a Q-Matrix to represent the learned values of state-action pairs.
class QLGraph {
public:
    using valid_actions_t = std::vector<size_t>;

    // Enum for reward constants used in learning.
    enum {
        NO_REWARD = 0,
        REWARD = 100,
        FORBIDDEN = -1
    };

    // Topology maps a state to a list of valid actions (next states).
    using Topology = std::unordered_map<size_t, std::list<size_t>>;

    // Constructor initializing the QLGraph with a given number of states, goal state, and topology.
    QLGraph(const size_t& n_of_states, const size_t& goal_state, const Topology& topology);

    // Constructor using a reward matrix.
    QLGraph(const QMatrix& reward_mtx)
        : _nOfStates(reward_mtx.size())
        , _rewardMtx(reward_mtx)
        , _q_mtx(reward_mtx.size())
    {
        assert(_nOfStates > 0);
    }

    void setLearningRate(const double& lr) noexcept { _learningRate = lr; }
    void setDiscountRate(const double& dr) noexcept { _discountRate = dr; }
    double getLearningRate() const noexcept { return _learningRate; }
    double getDiscountRate() const noexcept { return _discountRate; }

    // Helper class to assist in the learning process.
    struct Helper {
        Helper() noexcept
            : _rndGen(new RandomGenerator<>())
        {
        }

        virtual ~Helper() {};

        // Callbacks for the beginning and end of each episode.
        virtual void beginEpisode(const size_t& /*episode*/, QLGraph& /*qlobj*/) const {};
        virtual void endEpisode(const size_t& /*episode*/, QLGraph& /*qlobj*/) const {};

        // Check if a quit request is pending.
        virtual bool quitRequestPending() const { return false; }

        // Generate a random number.
        virtual double rnd() const noexcept { return (*_rndGen)(); }

    private:
        std::unique_ptr<RandomGenerator<>> _rndGen {};
    };

    // Function to start the learning process over a specified number of episodes.
    bool learn(const size_t& nOfEpisodes, const Helper& helper = Helper());

    // Returns the learned Q-Matrix.
    const QMatrix& get_q_mtx() const noexcept { return _q_mtx; }

    // Returns the next best state for a given state based on the Q-Matrix.
    size_t getNextStateFor(const size_t& state) const
    {
        return _q_mtx.maxarg(state);
    }

private:
    // Retrieves a list of valid actions for a given state.
    static valid_actions_t retrieveValidActions(const QMatrix& r, size_t state);

    // Selects a random action from a list of valid actions.
    size_t rand_of(const valid_actions_t& va)
    {
        assert(!va.empty());
        return va[rand() % va.size()];
    }

    size_t _nOfStates;
    size_t _goalState;

    QMatrix _rewardMtx; // Matrix representing the rewards for state transitions.
    QMatrix _q_mtx; // Q-Matrix representing the learned state-action values.

    double _learningRate { 0.8 }; // Learning rate in the Q-Learning algorithm.
    double _discountRate { 0.8 }; // Discount rate for future rewards.
};

}
