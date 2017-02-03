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

#ifndef __NU_GREEDY_POLICY_H__
#define __NU_GREEDY_POLICY_H__


/* -------------------------------------------------------------------------- */

#include "nu_random_gen.h"
#include <cmath>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template<class A, class AG, class T=double, class RG = random_gen_t<T>>
class greedy_policy_t {
public:
    using value_t = T;
    using action_t = A;
    using agent_t = AG;
    using rnd_gen_t = RG;   

    void set_epsilon(const value_t & e) noexcept {
        _epsilon = e;
    }

    value_t get_epsilon() const noexcept {
        return _epsilon;
    }

    template<class Q>
    action_t select_action(
        const agent_t& agent, 
        Q & q, 
        bool dont_explore = false) const
    {
        auto valid_actions = agent.valid_actions();

        assert(!valid_actions.empty());

        action_t action = valid_actions[0];

        value_t reward = 0;

        if (dont_explore || _rnd_gen() > get_epsilon()) {
            // get current agent state
            const auto & agent_state = agent.get_current_state();

            reward = q[agent_state][action];

            for (const auto & an_action : valid_actions) {
                const auto & val = q[agent_state][an_action];
                if (val > reward) {
                    reward = val;
                    action = an_action;
                }
            }
        }

        if (reward == .0) {
            action =
                valid_actions[size_t(_rnd_gen() *
                    value_t(valid_actions.size()))];
        }

        return action;
    }

    template<class Q>
    action_t get_learned_action(const agent_t& agent, Q & q) const {
        return select_action(agent, q, true);
    }

private:
    value_t _epsilon = .1;
    mutable rnd_gen_t _rnd_gen;

};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_GREEDY_POLICY_H__

