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

#ifndef __NU_SARSA_H__
#define __NU_SARSA_H__


/* -------------------------------------------------------------------------- */

#include "nu_learner_listener.h"

#include <unordered_map>


/* -------------------------------------------------------------------------- */

namespace nu {


/* -------------------------------------------------------------------------- */

template<
    class A,
    class S,
    class AG,
    class P,
    class T = double,
    class ARM = std::unordered_map<A, T>,
    class Q = std::unordered_map<S, ARM>>
class sarsa_t {
public:
    using real_t = double;
    using action_t = A;
    using state_t = S;
    using agent_t = AG;
    using policy_t = P;
    using reward_t = real_t;
    using listener_t = learner_listener_t<real_t>;

    using action_reward_t = ARM;
    using q_map_t = Q;

    sarsa_t(listener_t * listener = nullptr) noexcept 
        : _listener(listener) 
    {}
    
    sarsa_t(const sarsa_t&) = default;
    sarsa_t& operator=(const sarsa_t&) = default;

    real_t get_learning_rate() const noexcept {
        return _learning_rate;
    }

    real_t get_discount_rate() const noexcept {
        return _discount_rate;
    }

    void set_learning_rate(const real_t& lr) const noexcept {
        _learning_rate = lr;
    }

    void set_discount_rate(const real_t& dr) const noexcept {
        _discount_rate = dr;
    }

    action_t select_action(
        const agent_t& agent, 
        const policy_t & policy = policy_t()) 
    {
        return policy.template get_learned_action<q_map_t>(agent, get_q());
    }

    // learn episode
    real_t learn(agent_t& agent, const policy_t & policy=policy_t()) {

        size_t move_cnt = 0;

        action_t action = 
            policy.template select_action<q_map_t>(agent, get_q());
        
        auto state = agent.get_current_state();

        real_t reward = 0;

        while (!agent.goal() ) {
            if (_listener && !_listener->notify(reward, move_cnt++)) {
                break;
            }

            reward += update_q(agent, policy, state, action);
            ++move_cnt;
        }

        return reward;
    }

    const q_map_t & get_q() const noexcept {
        return _q_map;
    }
    
protected:
    q_map_t & get_q() noexcept {
        return _q_map;
    }  

    real_t update_q(
        agent_t& agent, 
        const policy_t & policy, 
        state_t & state,
        action_t & action) 
    {

        auto & qsa = get_q()[agent.get_current_state()][action];

        // update agent state
        agent.do_action(action); 

        // get current agent state
        const auto & state1 = agent.get_current_state();

        // get a reward for current state
        const auto reward = agent.reward(); 

        auto action1 = 
            policy.template select_action<q_map_t>(agent, get_q());

        qsa += 
            get_learning_rate() * 
            (reward + get_discount_rate() * get_q()[state1][action1] - qsa);

        state = state1;
        action = action1;

        return qsa;
    }

private:
    real_t _learning_rate = 0.1;
    real_t _discount_rate = 0.9;

    q_map_t _q_map;
    policy_t _policy;

    listener_t * _listener = nullptr;
};


/* -------------------------------------------------------------------------- */

}


/* -------------------------------------------------------------------------- */

#endif // __NU_SARSA_H__

