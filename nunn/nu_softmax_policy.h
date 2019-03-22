//
// This file is part of nunn Library
// Copyright (c) Antonino Calderone (antonino.calderone@gmail.com)
// All rights reserved.  
// Licensed under the MIT License. 
// See COPYING file in the project root for full license information.
//


/* -------------------------------------------------------------------------- */

#ifndef __NU_SOFTMAX_POLICY_H__
#define __NU_SOFTMAX_POLICY_H__


/* -------------------------------------------------------------------------- */

#include "nu_random_gen.h"
#include <cmath>
#include <cassert>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template<class A, class AG, class T = double, class RG = random_gen_t<T>>
class softmax_policy_t {
public:
    using real_t = T;
    using action_t = A;
    using agent_t = AG;
    using rnd_gen_t = RG;

    void set_temperature(const real_t & temperature) noexcept {
        _temperature = temperature;
    }

    real_t get_temperature() const noexcept {
        return _temperature;
    }

    template<class Q>
    action_t select_action(const agent_t& agent, Q & q) const
    {
        // Get agent to reward map
        auto action_reward = q[agent.get_current_state()];
        auto valid_actions = agent.valid_actions();
        decltype(action_reward) quasi_probs;
        
        real_t sum_reward = 0;

        for (const auto & item : valid_actions) {
            const auto reward = action_reward[item];
            const auto numerator = std::exp(reward / get_temperature());
            quasi_probs[item] = numerator;
            sum_reward += numerator;
        }

        assert(sum_reward != 0);

        // Select an action
        const auto cutoff = _rnd_gen();

        real_t sum = 0;
        auto it = quasi_probs.begin();

        for (; it != quasi_probs.end(); ++it) {

            const auto prob = it->second / sum_reward;
            sum += prob;

            if (sum > cutoff) {
                return it->first;
            }
        }

        assert(0);

        return it->first;
    }

    template<class Q>
    action_t get_learned_action(const agent_t& agent, Q & q) const {
        auto valid_actions = agent.valid_actions();

        assert(!valid_actions.empty());

        action_t action = valid_actions[0];

        real_t reward = 0;

        const auto & agent_state = agent.get_current_state();

        reward = q[agent_state][action];

        for (const auto & an_action : valid_actions) {
            const auto & val = q[agent_state][an_action];
            if (val > reward) {
                reward = val;
                action = an_action;
            }
        }

        if (reward == .0) {
            action = select_action(agent, q);
        }

        return action;
    }

private:
    real_t _temperature = 1.0;
    mutable random_gen_t<> _rnd_gen;
};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_SOFTMAX_POLICY_H__

