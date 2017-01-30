/*
*  This file is part of nunnlib
*
*  nunnlib is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  nunnlib is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with nunnlib; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  US
*
*  Author: Antonino Calderone <acaldmail@gmail.com>
*
*/


/* -------------------------------------------------------------------------- */

#include "nu_qlgraph.h"


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

qlearn_t::qlearn_t(
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

bool qlearn_t::learn(const size_t& n_of_episodes, cbk_t & cbk) 
{
    for (auto episode = 0; episode < n_of_episodes; ++episode) {

        cbk.begin_episode(episode, *this);

        if (cbk.quit_request_pending()) {
            return false;
        }

        auto current_state = cbk.rnd() % _n_of_states;

        bool goal = false;

        while (!goal) {

            if (cbk.quit_request_pending()) {
                return false;
            }

            auto valid_actions = retrieve_valid_actions(_reward_mtx, current_state);
            auto next_state = valid_actions[cbk.rnd() % valid_actions.size()];

            goal = _goal_state == current_state;

            auto & qsa = _q_mtx[current_state][next_state];
            auto & rsa = _reward_mtx[current_state][next_state];

            qsa +=
                _learning_rate *
                (rsa + _discount_rate * _q_mtx.max(next_state) - qsa);

            current_state = next_state;
        }

        cbk.end_episode(episode, *this);

        if (cbk.quit_request_pending()) {
            return false;
        }
    }

    _q_mtx.normalize();

    return true;
}


/* -------------------------------------------------------------------------- */

qlearn_t::valid_actions_t 
qlearn_t::retrieve_valid_actions(const qmtx_t& r, size_t state) 
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

