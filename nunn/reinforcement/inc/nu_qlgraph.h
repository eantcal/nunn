//
// This file is part of the nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.
// Licensed under the MIT License.
// See COPYING file in the project root for full license information.
//

#pragma once

#include "nu_qmtx.h"
#include "nu_random_gen.h"

#include <cassert>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace nu {

class QLGraph {
public:
    using valid_actions_t = std::vector<size_t>;

    enum {
        NO_REWARD = 0,
        REWARD = 100,
        FORBIDDEN = -1
    };

    // from   to
    using Topology = std::unordered_map<size_t, std::list<size_t>>;

    QLGraph(const size_t& n_of_states,
        const size_t& goal_state,
        const Topology& topology);

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

    struct Helper {
        Helper() noexcept
            : _rndGen(new RandomGenerator<>)
        {
        }
        virtual ~Helper() {};
        virtual void beginEpisode(const size_t& /*episode*/,
            QLGraph& /*qlobj*/) const {};
        virtual void endEpisode(const size_t& /*episode*/,
            QLGraph& /*qlobj*/) const {};
        virtual bool quitRequestPending() const { return false; }
        virtual double rnd() const noexcept { return (*_rndGen)(); }

    private:
        std::unique_ptr<RandomGenerator<>> _rndGen;
    };

    bool learn(const size_t& nOfEpisodes, const Helper& helper = Helper());

    const QMatrix& get_q_mtx() const noexcept { return _q_mtx; }

    size_t getNextStateFor(const size_t& state) const
    {
        return _q_mtx.maxarg(state);
    }

private:
    static valid_actions_t retrieveValidActions(const QMatrix& r, size_t state);

    size_t rand_of(const valid_actions_t& va)
    {
        assert(!va.empty());
        return va[rand() % va.size()];
    }

    size_t _nOfStates;
    size_t _goalState;

    QMatrix _rewardMtx;
    QMatrix _q_mtx;

    double _learningRate { 0.8 };
    double _discountRate  { 0.8 };
};

}
