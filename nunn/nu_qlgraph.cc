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

qlgraph_t::qlgraph_t(
        const size_t& n_of_states,
        const size_t& goal_state,
        const topology_t& topology) :
    _n_of_states(n_of_states),
    _goal_state(goal_state),
    _reward_mtx(n_of_states),
    _q_mtx(n_of_states)
{
    assert(goal_state < n_of_states);

    _reward_mtx.fill(FORBIDDEN);

    for (const auto & state : topology) {

        const auto & sl = state.first;

        for (auto & destination : state.second) {
            _reward_mtx[state.first][destination] =
                destination == _goal_state ? REWARD : NO_REWARD;
        }
    }
}


/* -------------------------------------------------------------------------- */

bool qlgraph_t::learn(const size_t& n_of_episodes, const helper_t & helper) 
{
    for (size_t episode = 0; episode < n_of_episodes; ++episode) {

        helper.begin_episode(episode, *this);

        if (helper.quit_request_pending()) {
            return false;
        }

        auto current_state = helper.rnd() % _n_of_states;

        bool goal = false;

        while (!goal) {

            if (helper.quit_request_pending()) {
                return false;
            }

            auto valid_actions = retrieve_valid_actions(_reward_mtx, current_state);
            auto next_state = valid_actions[helper.rnd() % valid_actions.size()];

            goal = _goal_state == current_state;

            auto & qsa = _q_mtx[current_state][next_state];
            auto & rsa = _reward_mtx[current_state][next_state];

            qsa +=
                _learning_rate *
                (rsa + _discount_rate * _q_mtx.max(next_state) - qsa);

            current_state = next_state;
        }

        helper.end_episode(episode, *this);

        if (helper.quit_request_pending()) {
            return false;
        }
    }

    _q_mtx.normalize();

    return true;
}


/* -------------------------------------------------------------------------- */

qlgraph_t::valid_actions_t 
qlgraph_t::retrieve_valid_actions(const qmtx_t& r, size_t state) 
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

