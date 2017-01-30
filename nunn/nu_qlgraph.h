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

#ifndef __NU_QLGRAPH_H__
#define __NU_QLGRAPH_H__


/* -------------------------------------------------------------------------- */ 

#include "nu_qmtx.h"

#include <vector>
#include <list>
#include <unordered_map>
#include <cassert>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

class qlgraph_t {
public:
    using valid_actions_t = std::vector<size_t>;

    enum { NO_REWARD = 0, REWARD = 100, FORBIDDEN = -1 };

                                            // from   to
    using topology_t = std::unordered_map< size_t, std::list<size_t> >;
    
    qlgraph_t( 
        const size_t& n_of_states, 
        const size_t& goal_state, 
        const topology_t& topology);

    qlgraph_t(const qmtx_t& reward_mtx) :
      _n_of_states(reward_mtx.size()),
      _reward_mtx(reward_mtx),
      _q_mtx(reward_mtx.size())
    {
        assert(_n_of_states>0);
    }

    void set_learning_rate(const double& lr) noexcept {
        _learning_rate = lr;
    }

    void set_discount_rate(const double& dr) noexcept {
        _discount_rate = dr;
    }

    double get_learning_rate() const noexcept {
        return _learning_rate;
    }

    double get_discount_rate() const noexcept {
        return _discount_rate;
    }

    struct helper_t {
        virtual ~helper_t() {};
        virtual void begin_episode(const size_t& /*episode*/, qlgraph_t & /*qlobj*/) const {};
        virtual void end_episode(const size_t& /*episode*/, qlgraph_t & /*qlobj*/) const {};
        virtual bool quit_request_pending() const { return false; }
        virtual size_t rnd() const noexcept { return size_t(rand()); }
    };

    bool learn(
        const size_t& n_of_episodes, 
        const helper_t & helper = helper_t());

    const qmtx_t& get_q_mtx() const noexcept {
        return _q_mtx;
    }

    const size_t get_next_state_for(const size_t& state) const {
        return _q_mtx.maxarg(state);
    }

private:
    static valid_actions_t retrieve_valid_actions(
        const qmtx_t& r, 
        size_t state);

    size_t rand_of(const valid_actions_t& va) {
        assert(!va.empty());
        return va[rand() % va.size()];
    }

    size_t _n_of_states;
    size_t _goal_state;

    qmtx_t _reward_mtx;
    qmtx_t _q_mtx;

    double _learning_rate = 0.8;
    double _discount_rate = 0.8;
};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_QLGRAPH_H__

