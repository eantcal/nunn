//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_E_GREEDY_POLICY_H__
#define __NU_E_GREEDY_POLICY_H__


/* -------------------------------------------------------------------------- */

#include "nu_random_gen.h"
#include <cmath>
#include <cassert>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template<class A, class AG, class T=double, class RG = random_gen_t<T>>
class e_greedy_policy_t {
public:
    using real_t = T;
    using action_t = A;
    using agent_t = AG;
    using rnd_gen_t = RG;   

    void set_epsilon(const real_t & e) noexcept {
        _epsilon = e;
    }

    real_t get_epsilon() const noexcept {
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

        real_t reward = 0;

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
                    real_t(valid_actions.size()))];
        }

        return action;
    }

    template<class Q>
    action_t get_learned_action(const agent_t& agent, Q & q) const {
        return select_action(agent, q, true);
    }

private:
    real_t _epsilon = .1;
    mutable rnd_gen_t _rnd_gen;

};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_E_GREEDY_POLICY_H__

