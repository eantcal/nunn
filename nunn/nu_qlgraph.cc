//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#include "nu_qlgraph.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

QLGraph::QLGraph(
        const size_t& n_of_states,
        const size_t& goal_state,
        const Topology& topology) :
    _n_of_states(n_of_states),
    _goalState(goal_state),
    _reward_mtx(n_of_states),
    _q_mtx(n_of_states)
{
    assert(goal_state < n_of_states);

    _reward_mtx.fill(FORBIDDEN);

    for (const auto & state : topology) {

        const auto & sl = state.first;

        for (auto & destination : state.second) {
            _reward_mtx[state.first][destination] =
                destination == _goalState ? REWARD : NO_REWARD;
        }
    }
}


/* -------------------------------------------------------------------------- */

bool QLGraph::learn(const size_t& nOfEpisodes, const Helper & helper) 
{
    for (size_t episode = 0; episode < nOfEpisodes; ++episode) {

        helper.beginEpisode(episode, *this);

        if (helper.quitRequestPending()) {
            return false;
        }

        auto current_state = helper.rnd() % _n_of_states;

        bool goal = false;

        while (!goal) {

            if (helper.quitRequestPending()) {
                return false;
            }

            auto validActions = retrieveValidActions(_reward_mtx, current_state);
            auto next_state = validActions[helper.rnd() % validActions.size()];

            goal = _goalState == current_state;

            auto & qsa = _q_mtx[current_state][next_state];
            auto & rsa = _reward_mtx[current_state][next_state];

            qsa +=
                _learningRate *
                (rsa + _discountRate * _q_mtx.max(next_state) - qsa);

            current_state = next_state;
        }

        helper.endEpisode(episode, *this);

        if (helper.quitRequestPending()) {
            return false;
        }
    }

    _q_mtx.normalize();

    return true;
}


/* -------------------------------------------------------------------------- */

QLGraph::valid_actions_t 
QLGraph::retrieveValidActions(const QMatrix& r, size_t state) 
{
    assert(state < r.size());

    valid_actions_t va;
    const auto & actions = r[state].to_stdvec();

    size_t idx = 0;

    for (const auto & action : actions) {
        if (action >= 0) {
            va.push_back(idx);
        }
        ++idx;
    }

    return va;
}


/* -------------------------------------------------------------------------- */

}

